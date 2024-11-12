###########################################################################################
# Example
# python normalization.py 'input directory' 'output file name' 'tile size' (default: 4096)
# python normalization.py /home/svs/ reinhardStats.csv 4096
###########################################################################################

import os
from multiprocessing import Pool, cpu_count
import sys
import openslide
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from color_conversion import lab_mean_std
from simple_mask import simple_mask


def color_normalization(file_path, stride_):
    print("Color normalization started for file:", file_path)
    wsi = openslide.OpenSlide(file_path)
    print("Opened slide file successfully")
    (lrWidth, lrHeight) = wsi.level_dimensions[0]
    array_x = np.arange(0, lrWidth + 1, stride_)
    array_y = np.arange(0, lrHeight + 1, stride_)
    mesh_x, mesh_y = np.meshgrid(array_x, array_y)

    sample_fraction = 0.01
    sample_pixels = []

    for i in range(mesh_x.shape[0] - 1):
        for j in range(mesh_x.shape[1] - 1):
            tile = wsi.read_region((int(mesh_x[i, j]), int(mesh_y[i, j])), 0, (stride_, stride_))#Reads a tile from the WSI at the given coordinates and size.
            tile = np.asarray(tile)
            tile = tile[:, :, :3]
            bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile, axis=2) > 250)#Counts the number of background pixels
            if (np.std(tile[:, :, 0]) + np.std(tile[:, :, 1]) + np.std(tile[:, :, 2])) / 3 > 18 \
                    and bn < stride_ * stride_ * 0.1:
                im_fgnd_mask_lres = simple_mask(tile)#Applies a simple mask to isolate foreground pixels.
                nz_ind = np.nonzero(im_fgnd_mask_lres.flatten())[0]# non-zero indices in the mask
                float_samples = sample_fraction * nz_ind.size
                num_samples = int(np.floor(float_samples))
                num_samples += np.random.binomial(1, float_samples - num_samples)
                sample_ind = np.random.choice(nz_ind, num_samples)#Randomly selects sample indices
                tile_pix_rgb = np.reshape(tile, (-1, 3))
                sample_pixels.append(tile_pix_rgb[sample_ind, :])

    sample_pixel = np.concatenate(sample_pixels, 0)
    sample_pixels_rgb = np.reshape(sample_pixel, (1, sample_pixel.shape[0], 3))
    print("Sample pixels shape:", sample_pixels_rgb.shape)
    print("Sample pixels min:", np.min(sample_pixels_rgb))
    print("Sample pixels max:", np.max(sample_pixels_rgb))
    mu, sigma = lab_mean_std(sample_pixels_rgb)
    print("Computed mean (mu):", mu)
    print("Computed standard deviation (sigma):", sigma)
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    src_mu_lab_out, src_sigma_lab_out = ReinhardStats(mu, sigma)
    print("Color normalization completed")
    return src_mu_lab_out, src_sigma_lab_out


def process_slide(slide_info):
    slide_path, stride = slide_info
    try:
        src_mu_lab, src_sigma_lab = color_normalization(slide_path, stride)
        return slide_path, src_mu_lab, src_sigma_lab
    except Exception as e:
        print(f"Error processing {slide_path}: {e}")
        return slide_path, None, None


def main():
    if len(sys.argv) != 4:
        print("Usage: ", sys.argv[0], "<path to the WSI directory> <path to the output file> <tile size>")
        exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    stride = int(sys.argv[3])
    print("Input directory:", input_dir)
    print("Output file:", output_file)
    print("Tile size:", stride)

    data = {
        "slidename": [], "mu1": [], "mu2": [], "mu3": [], "sigma1": [], "sigma2": [], "sigma3": []
    }
    df = pd.DataFrame(data)
   # Creates a list of tuples with slide paths and stride
    slide_paths = [(os.path.join(input_dir, img_name), stride) for img_name in sorted(os.listdir(input_dir))]
    with Pool(processes=cpu_count()) as pool:
        for slide_path, src_mu_lab, src_sigma_lab in tqdm(pool.imap_unordered(process_slide, slide_paths), total=len(slide_paths)):
            if src_mu_lab is not None and src_sigma_lab is not None:#Checks if the results are valid.
                df.loc[len(df.index)] = [
                    os.path.basename(slide_path),
                    src_mu_lab[0], src_mu_lab[1], src_mu_lab[2],
                    src_sigma_lab[0], src_sigma_lab[1], src_sigma_lab[2]
                ] #Appends the results to the DataFrame.
                print("Statistics computed for slide:", slide_path)
            else:
                print(f"Skipping slide {slide_path} due to errors.")

            df.to_csv(output_file, index=False)
            print("Output file saved:",output_file)


if __name__ == "__main__":
    main()
