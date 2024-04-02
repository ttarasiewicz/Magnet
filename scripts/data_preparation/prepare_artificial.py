import os
import random
import numpy as np
import cv2 as cv
from tqdm import tqdm
from scipy import ndimage
import imageio
import matplotlib.pyplot as plt
from shutil import copy2, copyfile
import json
import skimage.filters

hr_dir = 'SRR_benchmarks_org'

downscaling_factors = [3]
artificial_images = 9
interpolation = cv.INTER_CUBIC

max_translation = 0.95
noise_loc = 0.0
noise_scale = 0
contrast_scale = 0.0
brightness_scale = 0
blur_sigma = 0.0

interpolation_names = {cv.INTER_CUBIC:      ('cv.INTER_CUBIC', 'bicubic'),
                       cv.INTER_LINEAR:     ('cv.INTER_LINEAR', 'bilinear'),
                       cv.INTER_NEAREST:    ('cv.INTER_NEAREST', 'nearest'),
                       cv.INTER_LANCZOS4:   ('cv.INTER_LANCZOS4', 'lanczos'),
                       cv.INTER_AREA:       ('cv.INTER_AREA', 'area')}

interpolation_name, short_interp_name = interpolation_names[interpolation]
folder_name = f'SRR_benchmarks'

logging_variables = [downscaling_factors, artificial_images, blur_sigma, contrast_scale, brightness_scale,
                     max_translation, noise_loc, noise_scale, interpolation_name]

input_dir = r"..\..\datasets\\" + f"{hr_dir}"
output_dir = r"..\..\datasets\\" + f"{folder_name}"



def save_parameters_to_text_file(out_dir, local_vars):
    name = '_degradation_params.txt'
    file = open(os.path.join(out_dir, name), 'w')
    global_variables = globals().copy()
    for variable in logging_variables:
        for k, v in global_variables.items():
            if v is variable:
                file.write(f"{k} {v}\n")
                global_variables.pop(k)
                break
    for k, v in local_vars.items():
        file.write(f"{k} {str(v)}\n")
    file.close()


def save_translations(out_dir, examples):
    out_path = os.path.join(out_dir, 'translations.txt')
    with open(out_path, 'w') as json_file:
        json.dump(examples, json_file)


def show(*images):
    fig, axs = plt.subplots(1, len(images))
    axs = [axs] if not isinstance(axs, np.ndarray) else axs
    for i, im in enumerate(images):
        axs[i].imshow(im, cmap='gray')
    plt.show()

for subset in os.listdir(input_dir):
    subset_dir = os.path.join(input_dir, subset)
    datasets = os.listdir(subset_dir)
    for dataset in datasets:
        dataset_dir = os.path.join(subset_dir, dataset)
        examples = os.listdir(dataset_dir)
        for example in tqdm(examples, desc=f'{subset}: {dataset}'):
            example_path = os.path.join(dataset_dir, example)
            example_out_dir = os.path.join(output_dir, subset, dataset, example)
            os.makedirs(example_out_dir, exist_ok=True)
            hr_file = os.path.join(example_path, 'hr.png')
            copyfile(hr_file, os.path.join(example_out_dir, 'hr.png'))
            hr = cv.imread(hr_file, cv.IMREAD_UNCHANGED)
            translations = [(0., 0.)] + [(random.uniform(-max_translation, max_translation),
                                          random.uniform(-max_translation, max_translation)) for it in
                                         range(artificial_images - 1)]
            save_parameters_to_text_file(example_out_dir, {"translations": translations})
            for scale in downscaling_factors:
                lr_dir = os.path.join(example_out_dir, f'lr_{scale}x')
                os.makedirs(lr_dir, exist_ok=True)
                lr_translations = {}
                for iteration, (tr_x, tr_y) in enumerate(translations):
                    filename = f"lr_{str(iteration).zfill(2)}.png"
                    lr_translations[iteration] = (tr_x/scale, tr_y/scale)
                    shifted_hr = ndimage.shift(hr, shift=(tr_x, tr_y), mode='reflect')
                    lr_shape = tuple(map(lambda i: int(i / scale), hr.shape))
                    lr_array = cv.resize(shifted_hr, (lr_shape[1], lr_shape[0]), interpolation=interpolation)

                    # Preprocessing
                    noise = np.random.normal(loc=noise_loc, scale=noise_scale, size=lr_array.shape)
                    contrast_alpha = np.random.normal(loc=1.0, scale=contrast_scale, size=1)
                    brightness_beta = np.random.normal(loc=0.0, scale=brightness_scale, size=1)
                    augmented = contrast_alpha*lr_array + brightness_beta
                    blurred = skimage.filters.gaussian(augmented, sigma=blur_sigma, truncate=3.5)
                    noised = blurred + noise
                    final_lr = np.clip(lr_array, 0, 255).astype(np.uint8)
                    show(lr_array, augmented, blurred, noised)

                    # Save image
                    imageio.imwrite(os.path.join(lr_dir, filename), final_lr)
                save_translations(lr_dir, lr_translations)

