import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# Load the JSON file containing the list of TIFF names to generate
with open('./data_split.json', 'r') as jf:
    json_data = json.load(jf)

# Combine all image names from the split
tiff_list = json_data['train'] + json_data['valid'] + json_data['test'] + json_data['pretrain']

# Print the number of TIFF tiles to generate (for verification)
print(f'Number of TIFF tiles to generate: {len(tiff_list)}')

# Get list of .jp2 files in the folder
jp2_list = [x for x in os.listdir('./temp_raw_tiff_1') if x.lower().endswith('.jp2')]

print(f'Number of .jp2 files found: {len(jp2_list)}')
# Loop through each .jp2 file with a progress bar
with tqdm(total=len(jp2_list), unit='img') as pbar:
    print(f'Processing {len(jp2_list)} .jp2 files...')
    for jp2_name in jp2_list:
        # Load the .jp2 image as a NumPy array
        raw_tiff = np.array(Image.open(f'./temp_raw_tiff_1/{jp2_name}'))

        print(f'Processing {jp2_name} with shape {raw_tiff.shape}')
        ##################### Preprocess the image #####################


        # Check and print number of channels
        if raw_tiff.ndim == 3:
            print(f'{jp2_name} has {raw_tiff.shape[2]} channels')
            if raw_tiff.shape[2] == 4:

                #Replace the infrared channel with a synthetic one
                red = raw_tiff[:, :, 0].astype(np.float32)
                green = raw_tiff[:, :, 1].astype(np.float32)
                blue = raw_tiff[:, :, 2].astype(np.float32)

                synthetic_nir = (0.6 * red + 0.3 * green + 0.1 * blue).astype(np.uint8)
                raw_tiff[:, :, 3] = synthetic_nir  # Replace NIR with synthetic version
                
                # Alternatively, to silence the infrared channel, uncomment the following line:
                #raw_tiff[:, :, 3] = 0  # Silence the infrared channel


        ################################################################

        # Crop the image into 25 tiles of 1000x1000 pixels
        for ii in range(5):  # rows
            for jj in range(5):  # columns
                cropped_tiff_name = f'{jp2_name[:-4]}_{ii}{jj}'
                if cropped_tiff_name in tiff_list:
                    cropped = raw_tiff[1000*ii:1000*(ii+1), 1000*jj:1000*(jj+1)]
                    Image.fromarray(cropped).save(f'./cropped_tiff_1/{cropped_tiff_name}.tiff')

        pbar.update()



