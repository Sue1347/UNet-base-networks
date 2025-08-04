"""
Change the Original files to simple images.

According to the Segmentation, I will only use the slices, that have segmented Object to the Training Set.

And store the images into a directory

And also, the information for the images, the first one is the overall conclusion of the dataset: 
max, mean, median, min, percentile_00_5, percentile_99_5

And for the size of original images,
0, 1, 2, .. average shape

And the original records, for each file

And Spacing:
0,1,2,... average

And the original records

### common to clip the image using the 0.5 and 99.5 percentiles for better display or preprocessing

"""
import numpy as np
import pandas as pd
# import nrrd
import SimpleITK as sitk
import os
from scipy.stats import mode
import time
import csv
from PIL import Image

files_path = "/home/kevin/Downloads/Datasets/SPIDERDataset/data/training/" # images or labels
save_data_path = "/home/kevin/Documents/SUN Huajun/Projects/UNet-base-networks/data_perfusion/" # images or labels

pred_list = []
record_list = []
index = 0
# all_voxels = []
# Using memmap (example)
# data = np.memmap('images.dat', dtype='float32', mode='w+', shape=(200, 199, 400, 30)) 
start_time = time.time()

target_size = (256, 256)  # (width, height)


for file_name in os.listdir(os.path.join(files_path, "images")):
    
    file_path = os.path.join(files_path, "images",file_name)
    label_path = os.path.join(files_path, "labels",file_name.replace("t1", "seg"))
    print(f"File: {file_name}")
    
    image_3d = sitk.ReadImage(file_path) # 1_t1_0000.nii.gz
    data_index = file_name.split("_")[0]
    # print("The split index is:", data_index)

    spacing = image_3d.GetSpacing()  # spacing in [x, y, z]
    # origin = image_3d.GetOrigin()
    # direction = image_3d.GetDirection()
    image_np = sitk.GetArrayFromImage(image_3d)
    # print(image_np.min(), image_np.max())
    image_np = (image_np-image_np.min())/(image_np.max()-image_np.min())*255
    image_np = image_np.astype(np.uint8)

    label_3d = sitk.ReadImage(label_path) # 1_seg_0000.nii.gz
    label_np = sitk.GetArrayFromImage(label_3d)
    label_np = label_np * 255
    label_np = label_np.astype(np.uint8) 

    print(image_np.shape)
#     print(label_np.max())
    

    # all_voxels.extend(image_np.flatten())
    label_np[label_np == 6] = 0 # change to one label
    label_np[label_np > 0] = 1
    # print("the label number:", label_np.max())

    if image_np.shape[2] > image_np.shape[0]:
         print(f"doing transform for {data_index}")
         image_np = np.transpose(image_np, (1, 2, 0))
         label_np = np.transpose(label_np, (1, 2, 0))

    num_z = image_np.shape[2]
    for z in range(num_z):
        img = Image.fromarray(image_np[:,:,z])
        img_resized = img.resize(target_size, resample=Image.BILINEAR)
        lbl = Image.fromarray(label_np[:,:,z])
        lbl_resized = lbl.resize(target_size, resample=Image.BILINEAR)
        
        img_resized.save(f"{save_data_path}images/{data_index}_t1_{z}.jpg")
        lbl_resized.save(f"{save_data_path}labels/{data_index}_t1_{z}.gif")
        # print(f"saved one as: {data_index}_t1_{z} .jpg and .gif")

    pred_list.append(f"{data_index}_t1") # _{z}.jpg or .gif
    record_list.append(f"{data_index},{image_np.shape[0]},{image_np.shape[1]},{image_np.shape[2]},{spacing[0]},{spacing[1]},{spacing[2]},{np.min(image_np)},{np.max(image_np)},{np.mean(image_np)},{np.median(image_np)},{np.percentile(image_np, 0.5)}{np.percentile(image_np, 99.5)}")


# Then compute all the elements
# all_voxels = np.array(all_voxels)
# print("Min:", np.min(all_voxels))
# print("Max:", np.max(all_voxels))
# print("Mean:", np.mean(all_voxels))
# print("Median:", np.median(all_voxels))
# print("25%: ", np.percentile(all_voxels, 25))
# print("75%: ", np.percentile(all_voxels, 75))


with open(f"{save_data_path}List_data.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerows(pred_list)

with open(f"{save_data_path}List_records.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow([
        "Index", "DimX", "DimY", "DimZ",
        "SpacingX", "SpacingY", "SpacingZ",
        "Min", "Max", "Mean", "Median",
        "P0.5", "P99.5"
    ])
     # Write records
     wr.writerows(record_list)

# End timing
end_time = time.time()
# Calculate and print elapsed time
elapsed_time = end_time - start_time
print(f"Proprecess took {elapsed_time:.4f} seconds.")