# Written by Alen Joy

#To visualize the target images
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch

def image_grid(imgs, rows = None, cols = None, show_axes:bool = False):
    if (rows is None) != (cols is None):
        raise ValueError("Rows and columns need to be specified.")

    if rows is None:
        rows = len(imgs)
        cols = 1
    fig=plt.figure(figsize=(16, 16))
    
    for i in range(1, cols*rows+1):
        fig.add_subplot(rows, cols, i)
        imshow(torch.pow(imgs[i-1],1.0/2.2).cpu())        
        if not show_axes:
            plt.axis('off')
    fig.tight_layout()        
    plt.show()