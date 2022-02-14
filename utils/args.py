import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-height','--img_height', type = int, default = 576, help = "Please enter the image height")
    parser.add_argument('-width', '--img_width', type = int, default = 768, help = "Please enter the image width")
    parser.add_argument('-batch', '--batch_size', type = int, default = 20, help = "Please enter the batch size")
    parser.add_argument('-res','--resolution', type = int, default = 256, help = "Please enter the resolution")
    parser.add_argument('-f','--folder', type = str, default = 'folder', help = "Folder containing all the data")
    
    args = parser.parse_args()

    return args
