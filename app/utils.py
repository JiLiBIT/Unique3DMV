import torch
import numpy as np
from PIL import Image
import gc
import numpy as np
import numpy as np
from PIL import Image
from scripts.refine_lr_to_sr import run_sr_fast

GRADIO_CACHE = "/tmp/gradio/"

def clean_up():
    torch.cuda.empty_cache()
    gc.collect()

def remove_color(arr):
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # calc diffs
    base = arr[0, 0]
    diffs = np.abs(arr.astype(np.int32) - base.astype(np.int32)).sum(axis=-1)
    alpha = (diffs <= 80)
    
    arr[alpha] = 255
    alpha = ~alpha
    arr = np.concatenate([arr, alpha[..., None].astype(np.int32) * 255], axis=-1)
    return arr

def simple_remove(imgs, run_sr=True):
    """Only works for normal"""
    if not isinstance(imgs, list):
        imgs = [imgs]
        single_input = True
    else:
        single_input = False
    # if run_sr:
    #     imgs = run_sr_fast(imgs) #TODO
    rets = []
    for img in imgs:
        arr = np.array(img)
        arr = remove_color(arr)
        rets.append(Image.fromarray(arr.astype(np.uint8)))
    if single_input:
        return rets[0]
    return rets

def rgba_to_rgb(rgba: Image.Image, bkgd="WHITE"):
    new_image = Image.new("RGBA", rgba.size, bkgd)
    new_image.paste(rgba, (0, 0), rgba)
    new_image = new_image.convert('RGB')
    return new_image

def change_rgba_bg(rgba: Image.Image, bkgd="WHITE"):
    rgb_white = rgba_to_rgb(rgba, bkgd)
    new_rgba = Image.fromarray(np.concatenate([np.array(rgb_white), np.array(rgba)[:, :, 3:4]], axis=-1))
    return new_rgba

def split_image(image, rows=None, cols=None):
    """
        inverse function of make_image_grid
    """
    # image is in square
    if rows is None and cols is None:
        # image.size [W, H]
        rows = 1
        cols = image.size[0] // image.size[1]
        assert cols * image.size[1] == image.size[0]
        subimg_size = image.size[1]
    elif rows is None:
        subimg_size = image.size[0] // cols
        rows = image.size[1] // subimg_size
        assert rows * subimg_size == image.size[1]
    elif cols is None:
        subimg_size = image.size[1] // rows
        cols = image.size[0] // subimg_size
        assert cols * subimg_size == image.size[0]
    else:
        subimg_size = image.size[1] // rows
        assert cols * subimg_size == image.size[0]
    subimgs = []
    for i in range(rows):
        for j in range(cols):
            subimg = image.crop((j*subimg_size, i*subimg_size, (j+1)*subimg_size, (i+1)*subimg_size))
            subimgs.append(subimg)
    return subimgs

def make_image_grid(images, rows=None, cols=None, resize=None):
    if rows is None and cols is None:
        rows = 1
        cols = len(images)
    if rows is None:
        rows = len(images) // cols
        if len(images) % cols != 0:
            rows += 1
    if cols is None:
        cols = len(images) // rows
        if len(images) % rows != 0:
            cols += 1
    total_imgs = rows * cols
    if total_imgs > len(images):
        images += [Image.new(images[0].mode, images[0].size) for _ in range(total_imgs - len(images))]
    
    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new(images[0].mode, size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def extract_obj_and_resize_images(pil_list, target_size):
    resized_images = []
    box_sizes = []
    wheres = []
    for img in pil_list:
        np_img = np.array(img)[...,-1][...,None]
        non_black_pixels = np.where(np_img != 0)
        y_min, y_max = np.min(non_black_pixels[0]), np.max(non_black_pixels[0])
        x_min, x_max = np.min(non_black_pixels[1]), np.max(non_black_pixels[1])

        # is_nonzero_rgb = np.any(np_img[:, :, :3] != 0, axis=-1)
        # rows = np.any(is_nonzero_rgb, axis=1)
        # cols = np.any(is_nonzero_rgb, axis=0)
        # y_min, y_max = np.where(rows)[0][[0, -1]]
        # x_min, x_max = np.where(cols)[0][[0, -1]]

        cropped_img = img.crop((x_min, y_min, x_max, y_max))
        box_width = x_max - x_min
        box_height = y_max - y_min
        box_sizes.append((box_width, box_height))
        wheres.append((x_min, x_max, y_min, y_max))
        resized_img = cropped_img.resize(target_size, Image.LANCZOS)
        resized_images.append(resized_img)
    
    return resized_images, box_sizes, wheres

def extract_obj(pil_list):
    resized_images = []
    box_sizes = []
    wheres = []
    for img in pil_list:
        np_img = np.array(img)[...,-1][...,None]
        non_black_pixels = np.where(np_img != 0)
        y_min, y_max = np.min(non_black_pixels[0]), np.max(non_black_pixels[0])
        x_min, x_max = np.min(non_black_pixels[1]), np.max(non_black_pixels[1])

        # is_nonzero_rgb = np.any(np_img[:, :, :3] != 0, axis=-1)
        # rows = np.any(is_nonzero_rgb, axis=1)
        # cols = np.any(is_nonzero_rgb, axis=0)
        # y_min, y_max = np.where(rows)[0][[0, -1]]
        # x_min, x_max = np.where(cols)[0][[0, -1]]

        cropped_img = img.crop((x_min, y_min, x_max, y_max))
        box_width = x_max - x_min
        box_height = y_max - y_min
        box_sizes.append((box_width, box_height))
        wheres.append((x_min, x_max, y_min, y_max))
        # resized_img = cropped_img.resize(target_size, Image.LANCZOS)
        # resized_images.append(resized_img)
    
    return box_sizes, wheres


def resize_and_place_images(original_images, target_size, box_sizes, min_max_coords_list):
    new_images = []
    for original_image, box_size, min_max_coords in zip(original_images, box_sizes, min_max_coords_list):
        new_image = Image.new("RGBA", target_size, (0, 0, 0, 0))
        original_image = original_image.convert("RGBA")
        resized_image = original_image.resize(box_size, Image.LANCZOS)
        x_min, x_max, y_min, y_max = min_max_coords
        box_width, box_height = box_size
        x_offset = target_size[0] - x_min - box_width
        # y_offset = target_size[1] - y_min - box_height
        y_offset = y_min

        new_image.paste(resized_image, (x_offset, y_offset), resized_image)
        new_images.append(new_image)
    return new_images