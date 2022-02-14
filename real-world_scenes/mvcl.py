# Written by Alen Joy


import os
import sys
import glob
import time

import pyredner
import numpy as np
import torch
from tqdm import tqdm

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

from PIL import Image
from typing import Union

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())
pyredner.render_pytorch.print_timing = False

sys.path.append('../')
# import utils.args as args
from utils.args import get_args
from utils.folders import initialize_folder
from utils.blended_camera import read_cam
from utils.final_save import save_obj_file

def gamma_encode(image):
    return image ** (1.0/2.2)

# arg = args.get_args()
arg = get_args()

file_name = arg.folder
resolution = arg.resolution

camera_folder = os.path.join(arg.folder,'cams/')
imgs_folder = os.path.join(arg.folder,'images/')
obj_file = os.path.join(arg.folder,'geometry/mesh.obj')

result_dir,plt_dir,diffuse_dir,specular_dir,envmap_dir,material_map_dir,\
    optimized_results, final_results = initialize_folder(file_name)


camera = sorted(glob.glob(camera_folder+'0*'))
all_cams, fov = read_cam(camera)

cams = []
for i in range(len(all_cams)):
    cam = pyredner.Camera(cam_to_world = all_cams[i], fov = fov, resolution = (576,768))            
    cams.append(cam)

image_files = sorted(glob.glob(imgs_folder+'*.jpg'),key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
target_img = []
for i in range(len(image_files)):
    img_file = image_files[i]
    img_file = pyredner.imread(img_file)
    img_file = img_file.cuda(device = pyredner.get_device())
    target_img.append(img_file)


material_map, mesh_list, light_map = pyredner.load_obj(obj_file)
for _,mesh in mesh_list:
    mesh_normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)
    computed_uvs = pyredner.compute_uvs(mesh.vertices, mesh.indices)
uv_vertex, uv_index = computed_uvs


diffuse_tex = torch.tensor(\
    np.ones((resolution, resolution, 3), dtype=np.float32) * 0.0,
    requires_grad = True,
    device = pyredner.get_device())

specular_tex = torch.tensor(\
        np.ones((resolution,resolution,3),dtype=np.float32)*0.0,
        requires_grad = True,
        device = pyredner.get_device())

roughness_tex = torch.tensor(\
    np.ones((resolution, resolution, 1), dtype=np.float32) * 0.5,
    requires_grad = True,
    device = pyredner.get_device())

envmap_texels = torch.tensor(np.ones((32,64,3),dtype = np.float32)*0.01, requires_grad = True, device = pyredner.get_device())
envmap_predict = pyredner.EnvironmentMap(torch.abs(envmap_texels), directly_visible = True)


mat = pyredner.Material(diffuse_reflectance=pyredner.Texture(diffuse_tex), 
                       specular_reflectance=pyredner.Texture(specular_tex),
                       roughness=pyredner.Texture(roughness_tex))


objects = pyredner.Object(vertices = mesh.vertices,
                         indices = mesh.indices,
                         uvs = uv_vertex,
                         uv_indices = uv_index,
                         normals = mesh_normals,
                         material = mat)


optimizer = torch.optim.Adam([diffuse_tex,envmap_texels], lr = 1e-2, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=140, gamma=0.1)

n = 0
batch_size = arg.batch_size
losses = []

start = time.time()
for t in tqdm(range(1)):
    print('Diffuse Epoch:', t)
    optimizer.zero_grad()
    
    mat.diffuse_reflectance = pyredner.Texture(diffuse_tex)
    mat.specular_reflectance = pyredner.Texture(specular_tex)
    mat.roughness = pyredner.Texture(roughness_tex)

    envmap_predict = pyredner.EnvironmentMap(torch.abs(envmap_texels), directly_visible = True)
    
    gradients_diffuse = []

    gradients_diffuse_loss = 0
    total_img_loss = 0
    total_loss = 0
    
    for j in tqdm(np.random.permutation(len(cams)).tolist()[:batch_size]):
        scene = pyredner.Scene(cams[j], objects = [objects], envmap = envmap_predict)
        args = pyredner.RenderFunction.serialize_scene(\
                                                  scene = scene,
                                                  num_samples = 4, #16,4
                                                  max_bounces = 1)
        
        render = pyredner.RenderFunction.apply
        img = render(t+1, *args)
        
        pyredner.imwrite(img.cpu(),diffuse_dir+'iter_{}.png'.format(n))
        img_loss = (img - target_img[j]).pow(2).sum()
        img_loss.backward(retain_graph=True)
        gradients_diffuse.append(diffuse_tex.grad.detach().cpu().numpy())
        total_img_loss += img_loss
        n+=1
        
    total_img_loss = total_img_loss / batch_size

    magnitude_gradients_diffuse= np.linalg.norm(gradients_diffuse)
    if (magnitude_gradients_diffuse == 0.0):
        magnitude_gradients_diffuse = 1.0
    gradients_diffuse_loss = np.var(gradients_diffuse/magnitude_gradients_diffuse)    
    
    total_loss = total_img_loss * np.exp(gradients_diffuse_loss)

    print('Total Loss:', total_loss.item())
    print('Image loss:', total_img_loss.item())
        
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    diffuse_tex.data = diffuse_tex.data.clamp(0, 1)
    specular_tex.data = specular_tex.data.clamp(0, 1)
    roughness_tex.data = roughness_tex.data.clamp(1e-5, 1)
    
    losses.append(total_loss.item())

    print('Learning Rate:',optimizer.param_groups[0]['lr'])

plt.figure()
plt.plot(losses)
plt.savefig(plt_dir+'/diffuse_plt.png')


plt.figure(figsize=(30, 10))
plt.subplot(1, 6 ,1)
plt.plot(losses)
plt.subplot(1, 6, 2)
imshow(diffuse_tex.detach().cpu())
plt.axis('off')
plt.subplot(1, 6, 3)
imshow(specular_tex.detach().cpu())
plt.axis('off')
plt.subplot(1, 6, 4)
imshow(roughness_tex.detach().cpu())
plt.axis('off')
plt.subplot(1, 6 ,5)
imshow(gamma_encode(img).detach().cpu())
plt.axis('off')
plt.subplot(1,6,6)
imshow(envmap_texels.detach().cpu())
plt.axis('off')
plt.savefig(plt_dir+'diffuse_results.png')

# print('---------------Clearing CUDA memory-------------------')
# torch.cuda.empty_cache()

optimizer = torch.optim.Adam([diffuse_tex,specular_tex, roughness_tex,envmap_texels], lr = 0.005 , weight_decay = 1e-4 )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.5)

n = 0
batch_size = arg.batch_size
losses = []

start = time.time()
for t in tqdm(range(1)):
    print('Specular Epoch:', t)
    optimizer.zero_grad()
    
    mat.diffuse_reflectance = pyredner.Texture(diffuse_tex)
    mat.specular_reflectance = pyredner.Texture(specular_tex)
    mat.roughness = pyredner.Texture(roughness_tex)

    envmap_predict = pyredner.EnvironmentMap(torch.abs(envmap_texels), directly_visible = True)
    
    gradients_diffuse = []

    gradients_diffuse_loss = 0
    total_img_loss = 0
    total_loss = 0
    
    for j in tqdm(np.random.permutation(len(cams)).tolist()[:batch_size]):
        scene = pyredner.Scene(cams[j], objects = [objects], envmap = envmap_predict)
        args = pyredner.RenderFunction.serialize_scene(\
                                                  scene = scene,
                                                  num_samples = 4, #16,4
                                                  max_bounces = 1)
        
        render = pyredner.RenderFunction.apply
        img = render(t+1, *args)
        
        pyredner.imwrite(img.cpu(),specular_dir+'iter_{}.png'.format(n))
        img_loss = (img - target_img[j]).pow(2).sum()
        img_loss.backward(retain_graph=True)
        gradients_diffuse.append(diffuse_tex.grad.detach().cpu().numpy())
        total_img_loss += img_loss
        n+=1
        
    total_img_loss = total_img_loss / batch_size

    magnitude_gradients_diffuse= np.linalg.norm(gradients_diffuse)
    if (magnitude_gradients_diffuse == 0.0):
        magnitude_gradients_diffuse = 1.0
    gradients_diffuse_loss = np.var(gradients_diffuse/magnitude_gradients_diffuse)    
    
    total_loss = total_img_loss * np.exp(gradients_diffuse_loss)

    print('Total Loss:', total_loss.item())
    print('Image loss:', total_img_loss.item())
        
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    diffuse_tex.data = diffuse_tex.data.clamp(0, 1)
    specular_tex.data = specular_tex.data.clamp(0, 1)
    roughness_tex.data = roughness_tex.data.clamp(1e-5, 1)
    
    losses.append(total_loss.item())

    print('Learning Rate:',optimizer.param_groups[0]['lr'])

plt.figure()
plt.plot(losses)
plt.savefig(plt_dir+'/specular_plt.png')

plt.figure(figsize=(30, 10))
plt.subplot(1, 6 ,1)
plt.plot(losses)
plt.subplot(1, 6, 2)
imshow(diffuse_tex.detach().cpu())
plt.axis('off')
plt.subplot(1, 6, 3)
imshow(specular_tex.detach().cpu())
plt.axis('off')
plt.subplot(1, 6, 4)
imshow(roughness_tex.detach().cpu())
plt.axis('off')
plt.subplot(1, 6 ,5)
imshow(gamma_encode(img).detach().cpu())
plt.axis('off')
plt.subplot(1,6,6)
imshow(envmap_texels.detach().cpu())
plt.axis('off')
plt.savefig(plt_dir+'specular_results.png')

save_obj_file(objects,optimized_results+file_name)