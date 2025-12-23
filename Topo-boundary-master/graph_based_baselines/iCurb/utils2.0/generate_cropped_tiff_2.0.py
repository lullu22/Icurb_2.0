import os
import numpy as np
from PIL import Image
from tqdm import tqdm




# Folder containing RGB images
input_folder = '/home/c-lcuffaro/Desktop/prova'  # Change to your input folder path
output_folder = '/home/c-lcuffaro/Desktop/dataset_prova/cropped_tiff'  # Change to your output folder path
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist


# List of RGB images (.jpg or .png format)
rgb_list = [x for x in os.listdir(input_folder) if x.lower().endswith(('.jpg', '.png'))]

tile_size = 1000

# Loop over images
with tqdm(total=len(rgb_list), unit='img') as pbar:
    for rgb_name in rgb_list:
        
        # Load RGB image
        #rgb_image = np.array(Image.open(os.path.join(input_folder, rgb_name)))

        img_path = os.path.join(input_folder, rgb_name)
        img = Image.open(img_path)
        img = img.convert('RGB')
        rgb_image = np.array(img)

        # Check if image is RGB
        red = rgb_image[:, :, 0].astype(np.float32)
        green = rgb_image[:, :, 1].astype(np.float32)
        blue = rgb_image[:, :, 2].astype(np.float32)

        
        ###################################################### case with synthetic NIR ########################################################
        
            # Compute synthetic NIR channel
        synthetic_nir = (0.6 * red + 0.3 * green + 0.1 * blue).astype(np.uint8)

            # Stack RGB + NIR channels
        raw_tiff = np.dstack((rgb_image, synthetic_nir))

        h, w = raw_tiff.shape[0], raw_tiff.shape[1]

        ########################################################################################################################################

        ###################################################### case without synthetic NIR ######################################################
        
        #h, w = rgb_image.shape[0], rgb_image.shape[1] # we use only RGB channels

        ########################################################################################################################################
         
         
        num_tiles_y = (h + tile_size - 1) // tile_size
        num_tiles_x = (w + tile_size - 1) // tile_size

        for ii in range(num_tiles_y):
            for jj in range(num_tiles_x):
                y0 = ii * tile_size
                y1 = min(y0 + tile_size, h)
                x0 = jj * tile_size
                x1 = min(x0 + tile_size, w)

                # Include bottom edge
                if ii == num_tiles_y - 1:
                    y0 = max(h - tile_size, 0)
                    y1 = h

                # Include right edge
                if jj == num_tiles_x - 1:
                    x0 = max(w - tile_size, 0)
                    x1 = w
        ###################################################### case with synthetic NIR ########################################################
        
                cropped = raw_tiff[y0:y1, x0:x1]
                cropped_name = f'{rgb_name[:-4]}_{ii}_{jj}.tiff'
                #cropped_name = f'{rgb_name[:-4]}.tiff'

        ########################################################################################################################################

        ###################################################### case without synthetic NIR ######################################################
        
            #    cropped = rgb_image[y0:y1, x0:x1] # we use only RGB channels
            #    cropped_name = f'{rgb_name[:-4]}_{ii}_{jj}.png' # we use only RGB channels

        ########################################################################################################################################           
                    
                Image.fromarray(cropped).save(os.path.join(output_folder, cropped_name))

        pbar.update()

