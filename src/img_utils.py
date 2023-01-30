import os
from PIL import Image

from utils.helpers import colorize_mask


def center_crop(img, new_width=None, new_height=None):
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = img.crop((left, top, right, bottom))
    return im

def save_images(image, mask, output_path, image_file, palette):
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))
    
    colorized_mask = colorized_mask.convert('RGBA')
    
    colorized_mask.putalpha(127)

    image = image.convert('RGBA')

    output_im = Image.alpha_composite(image, colorized_mask)
    
    output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
