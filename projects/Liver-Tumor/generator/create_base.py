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
"""


# See : https://github.com/neheller/kits19/blob/master/starter_code/visualize.py

#DEFAULT_KIDNEY_COLOR = [255, 0, 0]
#DEFAULT_TUMOR_COLOR  = [0, 0, 255]
KIDNEY_COLOR = [255, 0, 0]
TUMOR_COLOR  = [0, 0, 255]


class ImageMaskDatasetGenerator:
  def __init__(self, resize=256):
    self.RESIZE = resize


  def resize_to_square(self, image):
     w, h  = image.size

     bigger = w
     if h > bigger:
       bigger = h
     pixel = image.getpixel((w-30, h-30))
     background = Image.new("RGB", (bigger, bigger), pixel)
    
     x = (bigger - w) // 2
     y = (bigger - h) // 2
     background.paste(image, (x, y))
     background = background.resize((self.RESIZE, self.RESIZE))

     return background
  

  
  def augment(self, image, output_dir, filename):
    # 2023/08/02
    #ANGLES = [30, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    ANGLES = [0, 90, 180, 270]
    for angle in ANGLES:
      rotated_image = image.rotate(angle)
      output_filename = "rotated_" + str(angle) + "_" + filename
      rotated_image_file = os.path.join(output_dir, output_filename)
      #cropped  =  self.crop_image(rotated_image)
      rotated_image.save(rotated_image_file)
      print("=== Saved {}".format(rotated_image_file))
      
    # Create mirrored image
    mirrored = ImageOps.mirror(image)
    output_filename = "mirrored_" + filename
    image_filepath = os.path.join(output_dir, output_filename)
    #cropped = self.crop_image(mirrored)
    
    mirrored.save(image_filepath)
    print("=== Saved {}".format(image_filepath))
        
    # Create flipped image
    flipped = ImageOps.flip(image)
    output_filename = "flipped_" + filename

    image_filepath = os.path.join(output_dir, output_filename)
    #cropped = self.crop_image(flipped)

    flipped.save(image_filepath)
    print("=== Saved {}".format(image_filepath))


  def create_mask_files(self, niigz, output_dir, index):
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
  
  def create_image_files(self, niigz, output_masks_dir, output_images_dir, index):
   
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
  

  def generate(self, data_dir, label_dir, output_images_dir, output_masks_dir):
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
        num_segmentations = self.create_mask_files(label_file,   output_masks_dir,  index)
        num_images        = self.create_image_files(image_file, output_masks_dir, output_images_dir, index)
        print(" image_nii_gz_file: {}  seg_nii_gz_file: {}".format(num_images, num_segmentations))

        if num_images != num_segmentations:
          raise Exception("Num images and segmentations are different ")
      else:
        print("Not found segmentation file {} corresponding to {}".format(seg_nii_gz_file, image_nii_gz_file))


if __name__ == "__main__":
  try:
    data_dir          = "./ImagesTr"
    label_dir         = "./LabelsTr"
    output_images_dir = "./Liver-master/images/"
    output_masks_dir  = "./Liver-master/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    generator = ImageMaskDatasetGenerator()
    generator.generate(data_dir, label_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


