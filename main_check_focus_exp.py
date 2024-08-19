import os
import cv2
import numpy as np
import re
import io
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift


# Find the focus image
def find_the_infocus_image(folder):
    image_files = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]
    max_contrast = 0
    max_contrast_file = None
    for image_file in image_files:
        img = cv2.imread(os.path.join(folder, image_file), cv2.IMREAD_GRAYSCALE)    
        contrast = img.std() / img.mean()  
        if contrast > max_contrast:
            max_contrast = contrast
            max_contrast_file = image_file
    print('Image with highest contrast:', max_contrast_file)
    print('Contrast:', max_contrast)
    return max_contrast_file

def exposure_time_array(folder):
    files = os.listdir(folder)
    exp_numbers = set()
    for file in files:
        # Use a regular expression to find the number part in the file name
        match = re.search('Exp(\d+)', file)
        if match:
            # If found, add it to the set of numbers
            exp_numbers.add(match.group(1))  # Save the string of digits
    exp_numbers = [int(num) for num in exp_numbers]
    exp_numbers = np.array(exp_numbers)
    exp_numbers_array = np.sort(exp_numbers)[::-1]
    print(f"exposure time: {exp_numbers_array}")
    return exp_numbers_array


# Get a list of all the files in the folder
################################
#change z format if necessary
################################
def z_plane(folder):
    files = os.listdir(folder)
    # Initialize a set to store the unique "_Z-XXXXXX" options
    z_options = set()
    
    for file in files:
        # Use a regular expression to find the "_Z-XXXXXX" part in the file name
        match = re.search('_Z(\d+\.\d+)', file)
        if match:
            # If found, add it to the set of options
            z_options.add(match.group(1))  # group(1) to get just the XXXXXX part
    
    z_options = [float(num) for num in z_options]
    z_options = np.array(z_options)
    # Sort the array from large to small
    z_options_array = np.sort(z_options)[::-1]
    # Print the unique "_Z-XXXXXX" options
    print(z_options_array)
    print(f"num of z planes: {z_options_array.size}")
    return z_options_array

import os
import shutil
import re

import os
import re

def find_closest_z_image(folder, target_z):
    # Initialize variables to store the best match
    closest_z = None
    closest_file = None
    min_diff = float('inf')  # Start with the largest possible difference
    file_names = os.listdir(folder)
    pattern = re.compile(r'Z(\d+\.\d+)')
    for file in file_names:
        match = pattern.search(file)
        if match:
            z_value = float(match.group(1))
            # Calculate the absolute difference from the target Z-value
            diff = abs(z_value - target_z)
            
            # Update the closest file if this file is closer to the target Z-value
            if diff < min_diff:
                min_diff = diff
                closest_z = z_value
                closest_file = file
    return os.path.join(folder, closest_file), closest_z

def fft_image(image, shift_back=False):
    from scipy.fftpack import fftshift, fft2
    img_float=image
    #f = np.fft.fft2(img_float) #mine, centered
    f = fftshift(fft2(img_float)) #julian
    magnitude_spectrum = (np.abs(f))**2
    return magnitude_spectrum

def fft2Dring(img, max_radius, ring_width):
    from scipy.fftpack import fftshift, fft2
    N, M = img.shape
    imgf = fftshift(fft2(img)) #julian
    #imgf = np.fft.fft2(img) #mine, centered
    imgfp = (np.abs(imgf) / (N * M)) ** 2  # Normalize
    dimDiff = abs(N - M)
    dimMax = max(N, M)
    # Make square
    if N > M:  # More rows than columns
        if dimDiff % 2 == 0:  # Even difference
            imgfp = np.pad(imgfp, ((0, 0), (dimDiff // 2, dimDiff // 2)), 'constant', constant_values=np.nan)
        else:  # Odd difference
            imgfp = np.pad(imgfp, ((0, 0), (dimDiff // 2, dimDiff // 2 + 1)), 'constant', constant_values=np.nan)
    elif N < M:  # More columns than rows
        if dimDiff % 2 == 0:  # Even difference
            imgfp = np.pad(imgfp, ((dimDiff // 2, dimDiff // 2), (0, 0)), 'constant', constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp, ((dimDiff // 2, dimDiff // 2 + 1), (0, 0)), 'constant', constant_values=np.nan)

    halfDim = dimMax // 2 + 1  # max r
    print(f"halfDim={halfDim}")
    # Compute radially average power spectrum
    X, Y = np.meshgrid(np.arange(-dimMax / 2, dimMax / 2), np.arange(-dimMax / 2, dimMax / 2))
    rho = np.sqrt(X**2 + Y**2)
    rho = np.round(rho).astype(int)
    Pf = np.zeros(halfDim)
    for r in range(halfDim):
        mask = (rho == r)
        Pf[r] = np.nanmean(imgfp[mask])
    return Pf

def single_image_folder(source_folder, target_folder, exposure_times, overexposure_files):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    best_images = {}
    pattern = re.compile(r'_Z([\d.]+)_Exp(\d+).png')
    for file in os.listdir(source_folder):
        match = pattern.search(file)
        if match:
            z_value = float(match.group(1))
            exposure_time = int(match.group(2))
            if file in overexposure_files:
                continue
            if z_value not in best_images or best_images[z_value][1] < exposure_time:
                best_images[z_value] = (file, exposure_time)

    for file, _ in best_images.values():
        shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))

    print(f"Single images have been copied to {target_folder}")


def overexposure_file(folder):
    files = os.listdir(folder)
    overexposed_files = []
    for file in files:
        file_path = os.path.join(folder, file)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # Read the image with original color depth
        if img is None:
            print(f"Failed to load image at {file_path}")
            continue
        if len(img.shape) == 2:  # Grayscale image
            # Check for overexposure in the grayscale image
            if np.sum(img > 254)>1000:
                overexposed_files.append(file)
                print(f"{file} is overexposed in grayscale.")
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image
            blue, green, red = cv2.split(img)
            if np.sum(red > 254)>1000 or np.sum(green > 254)>1000 or np.sum(blue > 254)>1000:
                overexposed_files.append(file)
                print(f"{file} is overexposed in color.")
    overexposed_files = np.array(overexposed_files)
    print(overexposed_files)
    return overexposed_files
    
# # Find the exposure files-- this is wrong way to check over-exposure for RGB image
# def overexposure_file(folder):
#     files = os.listdir(folder)
#     # Initialize a list to store the file names
#     file_names = []
#     # For each file
#     for file in files:
#         file_path = os.path.join(folder, file)
#         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#         # Check if the image is loaded properly
#         if img is None:
#             print(f"Failed to load image at {file_path}")
#             continue
#         # Check for overexposure
#         if img.max() > 254:
#             file_names.append(file)
#     file_names = np.array(file_names)
#     print(file_names)
#     return file_names

################################
#z scanning images, check the contrast
#change z format if necessary
import os
import cv2
import re
import numpy as np

import cv2
import numpy as np
import os
import re
from scipy.ndimage import convolve

def get_2d_gaussian(size, sigma):
    """Generate a 2D Gaussian kernel."""
    kernel = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    gaussian_kernel = np.outer(kernel, kernel.transpose())
    return gaussian_kernel

def get_filtered(image, low_pass=0, high_pass=100):
    """Apply low-pass and high-pass filters to the image."""
    if low_pass == 0:
        im_filtered_low = image
    else:
        size = min(max(int(3 * low_pass), 4), image.shape[1])
        gaussian_kernel = get_2d_gaussian(size, low_pass)
        im_filtered_low = convolve(image, gaussian_kernel, mode='reflect')
    
    if high_pass == 0:
        im_filtered = im_filtered_low
    else:
        size = min(int(2 * high_pass), im_filtered_low.shape[1])
        gaussian_kernel = get_2d_gaussian(size, high_pass)
        im_filtered_low_hp = convolve(im_filtered_low, gaussian_kernel, mode='reflect')
        im_filtered = im_filtered_low / (im_filtered_low_hp + 1e-5)  # Adding epsilon to avoid division by zero
    
    return im_filtered


def z_contrast(folder, color): #only use one color channel
    z = []
    contrast = []
    file_names = os.listdir(folder)
    pattern = re.compile(r'Z(\d+\.\d+)')  # Adjusted regex pattern to match your filename structure
    color_map = {'blue': 0,'green': 1,'red': 2}
    if color.lower() not in color_map:
        print(f"Error: '{color}' is not a valid color. Use 'blue', 'green', or 'red'.")
        return z, contrast
    channel_index = color_map[color.lower()]
    for file in file_names:
        match = pattern.search(file)
        if match:
            z_value = float(match.group(1))
            z.append(z_value)  # Append the extracted Z-value to the list
            file_path = os.path.join(folder, file)  # Corrected to use the full file path
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                channel = img[:, :, channel_index]
                if np.any(channel > 254):
                    print(f"image is overexposed in color channel={color}")
                contrast_value = channel.std() / channel.mean() # Calculate contrast for the specific channel
                contrast.append(contrast_value)
            else:
                print(f"Error: Image file {file_path} could not be read.")
        else:
            print(f"Warning: No Z-value found in filename {file}")
    print(f"len(z)={len(z)}")
    print(f"len(contrast)={len(contrast)}")
    return z, contrast

def z_contrast_with_f_filter(folder, low_pass=0, high_pass=100):
    z = []
    contrast = []
    file_names = os.listdir(folder)
    pattern = re.compile(r'Z(\d+\.\d+)')  # Adjusted regex pattern to match your filename structure
    for file in file_names:
        match = pattern.search(file)
        if match:
            z_value = float(match.group(1))
            file_path = os.path.join(folder, file)  # Corrected to use the full file path
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Check if the image is loaded properly
                # Apply filtering to remove low frequency noise
                filtered_img = get_filtered(img, low_pass, high_pass)
                if filtered_img.mean() != 0:  # Ensure the denominator is not zero
                    contrast_value = filtered_img.std() / filtered_img.mean()
                    z.append(z_value)
                    contrast.append(contrast_value)
                else:
                    print(f"Warning: Filtered image mean is zero, cannot compute contrast for {file_path}")
            else:
                print(f"Warning: Failed to load image {file_path}")
        else:
            print(f"Warning: No Z-value found in filename {file}")
    print(f"len(z)={len(z)}")
    print(f"len(contrast)={len(contrast)}")
    return z, contrast


# def z_contrast(input_path):
#     z = []
#     contrast = []
#     pattern = re.compile(r'Z(\d+\.\d+)')  # Adjusted regex pattern to match your filename structure

#     if os.path.isdir(input_path):  # Check if input is a folder
#         for file_name in os.listdir(input_path):
#             file_path = os.path.join(input_path, file_name)
#             match = pattern.search(file_name)
#             if match:
#                 z_value = float(match.group(1))
#                 img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#                 if img is not None:  # Check if the image is loaded properly
#                     if img.mean() != 0:  # Ensure the denominator is not zero
#                         contrast_value = img.std() / img.mean()
#                         z.append(z_value)
#                         contrast.append(contrast_value)
#                     else:
#                         print(f"Warning: Image mean is zero, cannot compute contrast for {file_path}")
#                 else:
#                     print(f"Warning: Failed to load image {file_path}")
#             else:
#                 print(f"Warning: No Z-value found in filename {file_path}")
#     elif os.path.isfile(input_path):  # Check if input is a file
#         file_name = os.path.basename(input_path)
#         match = pattern.search(file_name)
#         if match:
#             z_value = float(match.group(1))
#             img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#             if img is not None:  # Check if the image is loaded properly
#                 if img.mean() != 0:  # Ensure the denominator is not zero
#                     contrast_value = img.std() / img.mean()
#                     z.append(z_value)
#                     contrast.append(contrast_value)
#                 else:
#                     print(f"Warning: Image mean is zero, cannot compute contrast for {input_path}")
#             else:
#                 print(f"Warning: Failed to load image {input_path}")
#         else:
#             print(f"Warning: No Z-value found in filename {input_path}")

#     print(f"len(z)={len(z)}")
#     print(f"len(contrast)={len(contrast)}")
#     return z, contrast

###############################
#calculate contrast
###############################
def calculate_contrast(folder):
    contrasts = []
    filenames = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]
    for filename in filenames:
        # Load the image
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  
        # Calculate the contrast
        contrast = img.std() / img.mean()
        contrasts.append(contrast)
        print(f'{filename}: {round(contrast, 3)}')
    return contrasts


###############################
#calculate contrast
###############################
def calculate_contrast_middle300(folder):
    contrasts = []
    filenames = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]
    for filename in filenames:
        # Load the image
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  
        # Crop the image to the middle 100um*100um region
        # Assuming the entire image represents 1.1mm*1.1mm
        img_size = img.shape[0]  # Assuming the image is square
        crop_size = int(img_size * (300 / 1100))  # Size of the crop in pixels
        start = (img_size - crop_size) // 2  # Start of the crop
        end = start + crop_size  # End of the crop
        img = img[start:end, start:end]
        # Calculate the contrast
        contrast = img.std() / img.mean()
        contrasts.append(contrast)
        print(f'{filename}: {round(contrast, 3)}')
    return contrasts
    
######usage:
# folder = "/Users/qingjunwang/Documents/image files/superK"
# calculate_contrast(folder, filenames)    
#################
#normalization
##################

def save_gray_normalization(folder, ifsave, new_folder):
    filenames = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]
    if ifsave == 1 and not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  
        tenth_max = np.partition(img.flatten(), -10000)[-10000]
        img = (img / tenth_max) * 255
        # Save the image to the new folder
        if ifsave == 1:
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_gray_normalization{ext}"
            cv2.imwrite(os.path.join(new_folder, new_filename), img)
            print(f"Normalized file saved as {new_filename} in {new_folder}")
            print(f"{img.max()}")
            
##usage code:
# folder = "/Users/qingjunwang/Documents/image files/superK"
# new_folder="/Users/qingjunwang/Documents/image files/superK/normalization_velocity"
# ifsave=1
# main.save_gray_normalization(folder, image_files,ifsave,new_folder)


def normalize_max_255(img_path,color,image_save_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image at {path}")
    if len(img.shape) == 2:
        img1 = img.astype(np.float32)
    elif len(img.shape) == 3:
        color_map = {'red': 2, 'green': 1, 'blue': 0}
        if color in color_map:
            img1 = img[:, :, color_map[color]].astype(np.float32)
        else:
            print(f"Invalid color specified: {color}")
    else:
        print(f"Unsupported image format at {path}")
    max_value = img.max()
    scale = 255 / max_value
    scaled_img = img1 * scale
    contrast=scaled_img.std()/scaled_img.mean()
    vmin=0
    vmax=255
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    im1 = axs[0].imshow(img1, cmap="gray", vmin=vmin, vmax=vmax)
    axs[0].set_title("Original Image")
    fig.colorbar(im1, ax=axs[0])
    axs[0].grid(False)
    im2 = axs[1].imshow(scaled_img, cmap="gray", vmin=vmin, vmax=vmax)
    axs[1].set_title("scale of Images")
    fig.colorbar(im2, ax=axs[1])
    axs[1].grid(False)
    plt.show()
    cv2.imwrite(f'{image_save_path}_normalizedmax255_contrast{contrast}.png', scaled_img)
    print('scaled image saved')

def automatic_white_balance(img):
    # Calculate the mean of each channel
    b_mean, g_mean, r_mean = cv2.mean(img)[:3]
    # Calculate the overall mean
    mean = (r_mean + g_mean + b_mean) / 3
    # Scale the values of each channel
    img[:, :, 0] = np.clip((img[:, :, 0] * mean / b_mean), 0, 255)
    img[:, :, 1] = np.clip((img[:, :, 1] * mean / g_mean), 0, 255)
    img[:, :, 2] = np.clip((img[:, :, 2] * mean / r_mean), 0, 255)
    return img

def square_crop_img_upperleft(img,xrange):
    img_cropped = img[:xrange,:xrange]  # This will give you upperleft  region
    return img_cropped

def normalize_255(img_arr):
    max = img_arr.max()
    scale = 255 / max
    return (img_arr*scale).astype(np.uint8)
    
def RGB_combine_into_white(img_path_R,img_path_G,img_path_B,xrange,image_save_path):
    imgR = cv2.imread(img_path_R, cv2.IMREAD_COLOR)
    imgG = cv2.imread(img_path_G, cv2.IMREAD_COLOR)
    imgB = cv2.imread(img_path_B, cv2.IMREAD_COLOR)
    if imgR is None:
        print(f"Failed to load image at {img_path_R}")
    if imgG is None:
        print(f"Failed to load image at {img_path_G}")
    if imgB is None:
        print(f"Failed to load image at {img_path_B}")
        
    imgR_crop=square_crop_img_upperleft(imgR,xrange)
    imgG_crop=square_crop_img_upperleft(imgG,xrange)
    imgB_crop=square_crop_img_upperleft(imgB,xrange)

    imgR_crop_scale = normalize_255(imgR_crop)
    imgG_crop_scale = normalize_255(imgG_crop)
    imgB_crop_scale = normalize_255(imgB_crop)
    
    imgR_crop_singlechan = imgR_crop_scale[:, :, 2].astype(np.float32)
    imgG_crop_singlechan = imgG_crop_scale[:, :, 1].astype(np.float32)
    imgB_crop_singlechan = imgB_crop_scale[:, :, 0].astype(np.float32)

    height, width=imgR_crop_singlechan.shape
    img_combine = np.zeros((height, width, 3), dtype=np.uint8)
    img_combine[:, :, 0] = imgR_crop_singlechan
    img_combine[:, :, 1] = imgG_crop_singlechan
    img_combine[:, :, 2] = imgB_crop_singlechan
    img_combine_whitebalance=automatic_white_balance(img_combine)
    img_combine_whitebalance=normalize_255(img_combine_whitebalance)
    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
    axs[0].imshow(imgR_crop_scale); axs[0].set_title('Red image')
    axs[1].imshow(imgG_crop_scale); axs[1].set_title('Green image')
    axs[2].imshow(imgB_crop_scale); axs[2].set_title('Blue image')
    axs[3].imshow(img_combine_whitebalance); axs[3].set_title('Combined RGB Image')
    plt.show()
    cv2.imwrite(f'{image_save_path}/imgR_crop_scale.png', imgR_crop_scale)
    cv2.imwrite(f'{image_save_path}/imgG_crop_scale.png', imgG_crop_scale)
    cv2.imwrite(f'{image_save_path}/imgB_crop_scale.png', imgB_crop_scale)
    cv2.imwrite(f'{image_save_path}/img_combine_RGB.png', img_combine_whitebalance)
    print('scaled image saved')

def shaking_scan_amplitude(dic_images_key_path, amplitude_arr, image_pixel_per_um, select_color_chan, color, img_save_path):
    contrast={}
    radius_arr = np.ceil([amp * image_pixel_per_um for amp in amplitude_arr]) #in pixel
    sampling_arr = [np.ceil(amp/30*100) for amp in amplitude_arr] #1.8um per sampling on the trajectory
    for orginal_img_name, path in dic_images_key_path.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to load image at {path}")
            continue
        if len(img.shape) == 2:
            img1 = img.astype(np.float32)
        elif len(img.shape) == 3:
            if select_color_chan==1:
                color_map = {'red': 2, 'green': 1, 'blue': 0}
                if color in color_map:
                    img1 = img[:, :, color_map[color]].astype(np.float32)
                else:
                    print(f"Invalid color specified: {color}")
                    continue
            else:
                img1=img.astype(np.float32)
        else:
            print(f"Unsupported image format at {path}")
            continues

        for j in range(len(amplitude_arr)):
            if amplitude_arr[j]==0:
                avg_img=img1
            else:
                angle = np.linspace(0, 2 * np.pi, int(sampling_arr[j]))
                sum_img = np.zeros_like(img1)
                for a in angle:
                    x_shift = radius_arr[j] * np.cos(a)
                    y_shift = radius_arr[j] * np.sin(a)
                    if select_color_chan==1:
                        shifted_img = shift(img1, [y_shift, x_shift], mode="wrap")
                    else:
                        shifted_img = shift(img1, [y_shift, x_shift,0], mode="wrap")
                    sum_img += shifted_img
                avg_img = sum_img / len(angle)
            scaled_img = normalize_255(avg_img)
            contrast[(orginal_img_name,amplitude_arr[j])]=scaled_img.std()/scaled_img.mean()
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
            im1 = axs[0].imshow(img1.astype(np.uint8))
            axs[0].set_title("Original Image")
            axs[0].grid(False)
            im2 = axs[1].imshow(scaled_img)
            axs[1].set_title("Average of Images")
            axs[1].grid(False)
            plt.show();
            filename=f'{img_save_path}{orginal_img_name}_{int(sampling_arr[j])}points_{amplitude_arr[j]}um_contrast{contrast[(orginal_img_name,amplitude_arr[j])]}.png'
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            success = cv2.imwrite(filename, scaled_img)
            if success:
                print(f"Image successfully saved to {filename}")
            else:
                print(f"Failed to save image to {filename}")
    return contrast