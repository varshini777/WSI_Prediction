###################################################################################################
# Example
# python generate_mask_from_WSIs.py 'slide input directory' 'prediction output directory' 'model path' 'norm stats'
# python generate_mask_from_WSIs.py /home/svs/ /home/predict/ model.h5 reinhardStats.csv
###################################################################################################

import os
import sys
import math
import openslide
import numpy as np
import torch
import pandas as pd
from PIL import Image
from openslide.deepzoom import DeepZoomGenerator
from reinhard import reinhard
from torchvision import transforms
import cv2
from segmentation_models_pytorch import Unet
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_single_tile(file_path, m_model, s_mu_lab, s_sigma_lab, out_path, s_name, t_size=2048):
    try:
        # Open the slide
        slide = openslide.OpenSlide(file_path)
        
        # Create tile generator
        generator = DeepZoomGenerator(slide, tile_size=t_size, overlap=0, limit_bounds=True)
        
        # Get the highest zoom level
        highest_zoom_level = generator.level_count - 1
        
        # Set the level based on magnification
        try:
            mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            offset = math.floor((mag / 20) / 2)
            level = highest_zoom_level - offset
        except KeyError:
            level = highest_zoom_level - 1
        
        # Get a single tile from the center of the image
        cols, rows = generator.level_tiles[level]
        col = cols // 2
        row = rows // 2
        
        print(f"Accessing tile at level {level}, position ({col}, {row})")
        
        # Get the tile
        tile = np.array(generator.get_tile(level, (col, row)))
        
        if tile.shape[2] >= 3:
            tile = tile[:, :, :3]  # Take only RGB channels
            
            # Calculate quality metrics
            bn = np.sum(tile[:, :, 0] < 5) + np.sum(np.mean(tile, axis=2) > 245)
            std_threshold = np.mean([np.std(tile[:, :, i]) for i in range(3)])
            
            if std_threshold > 18 and bn < t_size * t_size * 0.3:
                # Reinhard normalization
                img_norm = reinhard(tile, reference_mu_lab, reference_std_lab, 
                                    src_mu=s_mu_lab, src_sigma=s_sigma_lab)
                
                # Convert to PIL Image
                img_pil = Image.fromarray((img_norm * 255).astype(np.uint8))
                
                # Define transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                
                # Process image in patches of 256x256
                patch_size = 256
                num_patches = t_size // patch_size
                full_prediction = np.zeros((t_size, t_size))
                
                for i in range(num_patches):
                    for j in range(num_patches):
                        # Extract patch
                        patch = img_norm[i*patch_size:(i+1)*patch_size, 
                                      j*patch_size:(j+1)*patch_size]
                        patch_pil = Image.fromarray((patch * 255).astype(np.uint8))
                        patch_tensor = transform(patch_pil).unsqueeze(0).to(device)
                        
                        # Model prediction
                        with torch.no_grad():
                            m_model.eval()
                            logits = m_model(patch_tensor)
                            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                        
                        # Store prediction
                        full_prediction[i*patch_size:(i+1)*patch_size, 
                                     j*patch_size:(j+1)*patch_size] = pred_prob
                
                # Convert prediction to grayscale format
                pred_mask_grayscale = (full_prediction * 255).astype(np.uint8)
                
                # Create a figure combining original image and mask
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                
                # Plot original image
                ax1.imshow(tile)
                ax1.set_title('Original WSI Tile')
                ax1.axis('off')
                
                # Plot prediction mask
                ax2.imshow(pred_mask_grayscale, cmap='gray')
                ax2.set_title('Prediction Mask')
                ax2.axis('off')
                
                # Save combined figure
                combined_path = os.path.join(out_path, f"{s_name}_combined_result.png")
                plt.savefig(combined_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                print(f"Saved combined result to {combined_path}")
                
            else:
                print(f"Tile rejected due to quality thresholds (std: {std_threshold:.2f}, bright/dark pixels: {bn})")
        else:
            print(f"Invalid tile format: shape = {tile.shape}")
            
    except Exception as e:
        print(f"Error processing slide {s_name}: {str(e)}")

def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <WSI_directory> <output_directory> <model_path> <normalization_stats>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = sys.argv[3]
    norm_stats = sys.argv[4]

    os.makedirs(output_dir, exist_ok=True)

    # Reference LAB color normalization stats
    global reference_mu_lab, reference_std_lab
    reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    reference_std_lab = [0.57506023, 0.10403329, 0.01364062]
    
    print("Loading model...")
    try:
        # Load model
        model = Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            classes=1,
            activation=None
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Load normalization data
    try:
        df = pd.read_csv(norm_stats)
    except Exception as e:
        print(f"Error loading normalization stats: {str(e)}")
        sys.exit(1)
    
    # Process each slide
    whole_slide_images = sorted([f for f in os.listdir(input_dir) if f.endswith('.svs')])
    for img_name in whole_slide_images:
        print(f"Processing slide: {img_name}")
        src_df = df.loc[df['slidename'] == img_name].to_numpy()[:, 1:].astype(np.float64)
        
        if len(src_df) == 0:
            print(f"No normalization data found for {img_name}")
            continue
        
        src_mu_lab = src_df[0, :3]
        src_sigma_lab = src_df[0, 3:]
        
        slide_path = os.path.join(input_dir, img_name)
        process_single_tile(slide_path, model, src_mu_lab, src_sigma_lab, output_dir, img_name.split('.')[0])

if __name__ == "__main__":
    main()
