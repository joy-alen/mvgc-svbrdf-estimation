import numpy as np
import torch

def read_cam(camera):
    cam_files = []
    for c in camera:
        with open(c) as file:
            read_files = file.readlines()
            loaded_files = []
            read_files = [x.strip() for x in read_files]
            
            r1 = read_files[1].split(' ')
            r2 = read_files[2].split(' ')
            r3 = read_files[3].split(' ')
            r4 = read_files[4].split(' ')
            r7 = read_files[7].split(' ')
            loaded_files.append([r1,r2,r3,r4])
            
            loaded_files = np.asarray(loaded_files, dtype = np.float32)
            
            cam_file = []
            for i in loaded_files:
                val = np.linalg.inv(i)
                val[:,1] *= -1
                cam_file.append(val)
            cam_files.append(np.squeeze(np.asarray(cam_file)))
    cam_files = np.asarray(cam_files, dtype = np.float32)
    cam_files = torch.from_numpy(cam_files)

    f = np.asarray(r7[0], dtype = np.float32)
    fov = np.asarray(2 * np.degrees(np.arctan((0.5 * 768)/f)), dtype = np.float32)
    fov = torch.tensor(fov).unsqueeze(0)

    return cam_files, fov