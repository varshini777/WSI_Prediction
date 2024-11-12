# WSI Prediction - Tumor, Stroma, and TIL Regions Segmentation

This project provides tools for whole slide image (WSI) analysis and segmentation of tumor, stroma, and tumor-infiltrating lymphocyte (TIL) regions using deep learning models such as U-Net and DeepLabV3. It involves two main steps:
1. **Color Normalization**: The `normalization.py` script calculates color statistics (mean and standard deviation in the LAB color space) for each WSI and stores them in a CSV file. These statistics are later used for color normalization during mask generation.
2. **Segmentation Mask Generation**: The `generate_mask_from_wsi.py` script processes WSIs using pre-trained models (U-Net or DeepLabV3) and generates segmentation masks (in PNG format) for tumor, stroma, and TIL regions.

## Table of Contents
1. [Color Normalization Step](#color-normalization-step)
2. [Segmentation Mask Generation](#segmentation-mask-generation)
3. [Models Used](#models-used)
4. [Results](#results)

Color Normalization Step
The color normalization step is essential for improving the consistency of images from different sources. The normalization.py script computes the mean and standard deviation values of the LAB color channels (L, A, B) for each slide. These values are used to normalize the colors in the WSI during mask generation, ensuring that the model performs robustly on diverse inputs.

Run the script as follows:

bash
Copy code
python normalization.py /path/to/wsi_files /path/to/output_color_stats.csv

Color Normalization Process:
The script reads each WSI.
Converts the image to the LAB color space.
Calculates the mean and standard deviation for each color channel.
Outputs the results in a CSV file.

Segmentation Mask Generation
In the mask generation step, the generate_mask_from_wsi.py script performs segmentation using a pre-trained deep learning model. It uses the color normalization statistics computed earlier to adjust the color balance of the image before performing segmentation. The segmentation masks are saved as PNG images.

Run the script as follows:

bash
Copy code
python generate_mask_from_wsi.py /path/to/wsi_files /path/to/output_masks /path/to/model_weights.pth /path/to/output_color_stats.csv

Segmentation Process:
The script loads the WSI.
Applies color normalization based on the CSV statistics.
Loads the pre-trained model (U-Net or DeepLabV3).
Performs segmentation to detect tumor, stroma, and TIL regions.
Saves the output mask as a PNG file.

Models Used

U-Net: A convolutional neural network designed for biomedical image segmentation. It is effective for pixel-wise classification and is used in this project for segmenting tumor, stroma, and TIL regions.
DeepLabV3: A state-of-the-art deep learning model for semantic image segmentation. DeepLabV3 uses atrous convolution to capture multi-scale context information, which helps in accurate segmentation of regions.
The models are trained on relevant datasets and fine-tuned to handle WSI data effectively. Pre-trained model weights can be used, or you can train your own models if required.

Results

The output of the segmentation process is a mask image in PNG format for each WSI, which highlights the tumor, stroma, and TIL regions. These masks can be used for further analysis, visualization, or quantitative studies.
