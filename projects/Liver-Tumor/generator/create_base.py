# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import shutil
import cv2

import glob
import numpy as np
import math
import nibabel as nib
import traceback

# Read file
"""
scan = nib.load('/path/to/stackOfimages.nii.gz')
# Get raw data
scan = scan.get_fdata()
print(scan.shape)
(num, width, height)
or 
(width, height, num)
"""


def create_mask_files(niigz, output_dir, index):
    print("--- niigz {}".format(niigz))
    nii = nib.load(niigz)

    print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    #data = np.asanyarray(nii.dataobj)
    num_images = data.shape[2] # math.floor(data.shape[2]/2)
    print("--- num_images {}".format(num_images))

    num = 0
    for i in range(num_images):
      img = data[:, :, i]

      if np.any(img > 0):
        img = img*255
        filepath = os.path.join(output_dir, str(index) + "_" + str(i) + ".jpg")
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        num += 1
    return num
  
def create_image_files(niigz, output_masks_dir, output_images_dir, index):
   
    print("--- create_image_files niigz {}".format(niigz))
    nii = nib.load(niigz)

    print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    #data = np.asanyarray(nii.dataobj)
    num_images = data.shape[2] # math.floor(data.shape[2]/2)
    print("--- num_images {}".format(num_images))
    num = 0
    for i in range(num_images):
      img = data[:, :, i]
   
      filename = str(index) + "_" + str(i) + ".jpg"
      mask_filepath = os.path.join(output_masks_dir, filename)
      if os.path.exists(mask_filepath):
        filepath = os.path.join(output_images_dir, filename)
   
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        num += 1
    return num
  

def generate(data_dir, label_dir, output_images_dir, output_masks_dir):
    #    data_dir          = "./data/case_*"
    image_files = glob.glob(data_dir + "/*.nii")
    
    print("--- num dirs {}".format(len(image_files)))

    index = 10000
    for image_file in image_files:
      print("== image_file {}".format(image_file))
      basename  = os.path.basename(image_file)
      label_file       =  os.path.join(label_dir, basename)
      index += 1
      if os.path.exists(image_file) and os.path.exists(label_file):
        num_segmentations = create_mask_files(label_file,   output_masks_dir,  index)
        num_images        = create_image_files(image_file, output_masks_dir, output_images_dir, index)
        print(" image_nii_gz_file: {}  seg_nii_gz_file: {}".format(num_images, num_segmentations))

        if num_images != num_segmentations:
          raise Exception("Num images and segmentations are different ")
      else:
        print("Not found segmentation file {} corresponding to {}".format(seg_nii_gz_file, image_nii_gz_file))

"""
./Task03_Liver_rs
+-- ImagesTr
+-- LabelsTr

"""
if __name__ == "__main__":
  try:
    data_dir          = "./ImagesTr"
    label_dir         = "./LabelsTr"
    output_images_dir = "./Liver-base/images/"
    output_masks_dir  = "./Liver-base/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    generate(data_dir, label_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


