import os
import sys
def initialize_folder(folder):

    result_dir = os.path.join(folder,'results/')
    if(os.path.isdir(result_dir)!= True):
        os.makedirs(result_dir)

    plt_dir = os.path.join(folder,'results/plots/')
    if(os.path.isdir(plt_dir)!= True):
        os.mkdir(plt_dir)

    diffuse_dir = os.path.join(folder,'results/diffuse_optimization/')
    if(os.path.isdir(diffuse_dir)!=True):
        os.mkdir(diffuse_dir)

    specular_dir = os.path.join(folder,'results/specular_optimization/')
    if(os.path.isdir(specular_dir)!=True):
        os.mkdir(specular_dir)

    envmap_dir = os.path.join(folder,'results/envmap_optimization/')
    if(os.path.isdir(envmap_dir)!=True):
        os.mkdir(envmap_dir)

    material_map_dir = os.path.join(folder,'results/material_maps/')
    if(os.path.isdir(material_map_dir)!=True):
        os.mkdir(material_map_dir)

    optimized_results = os.path.join(folder,'results/optimized/')
    if(os.path.isdir(optimized_results)!=True):
        os.mkdir(optimized_results)

    final_results = os.path.join(folder,'results/final/')
    if(os.path.isdir(final_results)!=True):
        os.mkdir(final_results)

    return result_dir, plt_dir, diffuse_dir, specular_dir, envmap_dir, material_map_dir, optimized_results,final_results