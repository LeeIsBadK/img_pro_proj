# main.py
import numpy as np
import os
import cv2
from config import *
from preprocess import mask_image, segment_and_threshold, sharpen_image, apply_power_rule, apply_average_filter
from segmentation import get_connected_component, height_segment_threshold
from utils import create_directory
from region_grow import region_grow_cont, region_split_merge, merge_small_region
from colormap import get_distinct_colormap, apply_colormap_to_component

def main(image_path):
    # Ensure the output directory exists
    create_directory(OUTPUT_DIR)

    # Step 1: Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    image_path = os.path.splitext(os.path.basename(image_path))[0]

    # Step 2: Preprocess the image
    img1 = mask_image(img, MASK_LEFT, MASK_RIGHT)
    img1 = apply_power_rule(img1, GAMMA)
    img2 = segment_and_threshold(img1, NUM_SEGMENTS, THRESHOLD_PERCENT)
    img3 = sharpen_image(img1)


    # AVG Filter
    avg1 = apply_average_filter(img1, 3)
    avg2 = apply_average_filter(avg1, 3)
    avg3 = apply_average_filter(avg2, 3)
    avg4 = apply_average_filter(avg3, 3)

    # height_segment_threshold
    mask1 = height_segment_threshold(avg1,16,3)
    mask2 = height_segment_threshold(avg2,16,3)
    mask3 = height_segment_threshold(avg3,16,3)

    bitwisefilter = cv2.bitwise_and(mask1, mask2)
    bitwisefilter = cv2.bitwise_and(bitwisefilter, mask3)

    # Connected Component
    component = get_connected_component(bitwisefilter)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_img4_connected_component.jpg'), component)

    # region growing
    region = region_grow_cont(img3, component, 28, vis=False)
    merge1 = region_split_merge(region)
    merge2 = merge_small_region(merge1, 0.5)
    merge3 = merge_small_region(merge2, 0.5)
    merge4 = merge_small_region(merge3, 0.5)
    merge4 = merge_small_region(merge4, 0.7)

    # print the number of connected components
    for i in range(1, np.max(merge4) + 1):
        print(f"Connected component {i}: {np.sum(merge3 == i)} pixels")
        # save the connected component as an image and mark as white
        if i < len(LABEL):
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_{LABEL[i]}.jpg'), np.where(merge4 == i, 255, 0).astype(np.uint8))
        else:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_{i}.jpg'), np.where(merge4 == i, 255, 0).astype(np.uint8))

        


        # # crop the connected component from the original image
        x, y, w, h = cv2.boundingRect(np.where(merge4 == i, 255, 0).astype(np.uint8))
        # cv2.imwrite(os.path.join(OUTPUT_DIR, f'connected_component_{i}_crop.png'), img[y:y+h, x:x+w])

        # # highlight the connected component in the original image
        # img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

        #hightlight the connected component in the original image with red color
        img_rec = img.copy()
        img_rec = cv2.rectangle(cv2.cvtColor(img_rec, cv2.COLOR_GRAY2BGR), (x-10, y-10), (x+w+10, y+h+10), (0, 0, 255), 2)
        if i < len(LABEL):
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_{LABEL[i]}_highlight.jpg'), img_rec)
        else:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_{i}_highlight.jpg'), img_rec)

        
    img_cmap = apply_colormap_to_component(merge4)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_img5_connected_component_cmap.jpg'), img_cmap)

    # get cmapped image with original gray image img1 to shap 3 channel image and plus
    img6 = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.5, img_cmap, 0.5, 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_img6_connected_component_on_gray.jpg'), img6)

    # Step 3: Save preprocessed images
    # save_spikes_as_png(region, OUTPUT_DIR)  # Save individual regions as separate .png files

    # Other image saving steps
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_img1_masked.jpg'), img1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_img2_thresholded.jpg'), img2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{image_path}_img3_sharpened.jpg'), img3)
    #cv2.imwrite(os.path.join(OUTPUT_DIR, 'img4_masked_sharpened.jpg'), img4)


if __name__ == "__main__":
    # Replace with the path to your X-ray image
    image_path = os.path.join(DATA_DIR, '3.jpeg')
    main(image_path)
