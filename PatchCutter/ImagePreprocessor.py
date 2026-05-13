import os
from PIL import Image
import numpy as np
import math

class ImagePreprocessor:
    def __init__(self, image: Image, overlapping_tolerance = 0.2, num_layers = 3, min_patch_resolution = 256, max_patch_resolution = 2048) -> None:
        self.image = image
        self.overlapping_tolerance = overlapping_tolerance
        self.num_layers = num_layers
        self.image_layers = [{} for _ in range(self.num_layers)]
        self.min_patch_resolution = min_patch_resolution
        self.max_patch_resolution = max_patch_resolution
        return
    
    # Crop the image into m times m patches and allow overlapping
    def crop_by_size(self, p_size: int):
        image_patches = []
        img_w, img_h = self.image.size
        
        # m = ceil((L - P) / (P * (1 - overlap))) + 1
        def calc_m_and_stride(length, patch_size, overlap_tol):
            if patch_size >= length:
                return 1, 0
            m = math.ceil((length - patch_size) / (patch_size * (1 - overlap_tol))) + 1
            stride = (length - patch_size) / (m - 1)
            return m, stride

        m_w, stride_w = calc_m_and_stride(img_w, p_size, self.overlapping_tolerance)
        m_h, stride_h = calc_m_and_stride(img_h, p_size, self.overlapping_tolerance)

        print(f"Level Patch Size: {p_size}, Grid: {m_w}x{m_h}")

        for i in range(m_h):
            for j in range(m_w):
                offset_y = int(i * stride_h)
                offset_x = int(j * stride_w)
                
                patch = self.image.crop((offset_x, offset_y, offset_x + p_size, offset_y + p_size))
                
                image_patches.append({
                    'image': patch, 
                    'offset_x': offset_x, 
                    'offset_y': offset_y
                })
        return image_patches

    def process(self):
        if self.num_layers <= 1:
            patch_sizes = [self.min_patch_resolution]
        else:
            patch_sizes = np.linspace(self.min_patch_resolution, self.max_patch_resolution, self.num_layers, dtype=int)
        
        print("Generated layer sizes:", patch_sizes)

        for i, p_size in enumerate(patch_sizes):
            self.image_layers[i] = self.crop_by_size(p_size)

        return self.image_layers

    def save_patches(self, folder_path = r'.', layer = 0, clear_folder = True):
        # Create folder if not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Clear folder if required
        if clear_folder:
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))

        for patch in self.image_layers[layer]:
            image = patch['image']
            offset_x = patch['offset_x']
            offset_y = patch['offset_y']
            image.save(os.path.join(folder_path, f'{layer}_{offset_y}_{offset_x}.png'))
        return
    
    def get_image_patches(self, layer):
        offset_xs = [p['offset_x'] for p in self.image_layers[layer]]
        offset_ys = [p['offset_y'] for p in self.image_layers[layer]]
        images = [p['image'] for p in self.image_layers[layer]]

        return images, offset_xs, offset_ys
