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
from tkinter import Image
import cv2

import glob
import numpy as np
import math
import nibabel as nib
import traceback
from PIL import Image, ImageOps

# Read file
"""
scan = nib.load('/path/to/stackOfimages.nii.gz')
# Get raw data
scan = scan.get_fdata()
print(scan.shape)
(num, width, height)
"""

class ImageMaskDatasetGenerator:
  def __init__(self, resize=256, rotation=False):
    self.RESIZE = resize
    self.ROTATION = rotation

  def resize_to_square(self, image):
     w, h  = image.size

     bigger = w
     if h > bigger:
       bigger = h
     pixel = image.getpixel((w-10, h-10))
     pixel =(0, 0, 0)
     background = Image.new("RGB", (bigger, bigger), pixel)
    
     x = (bigger - w) // 2
     y = (bigger - h) // 2
     background.paste(image, (x, y))
     background = background.resize((self.RESIZE, self.RESIZE))

     return background

  
  def augment(self, image, output_dir, filename):
    # 2023/08/06
    #ANGLES = [30, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    if self.ROTATION:
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


  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
    #    data_dir          = "./data/case_*"
    image_files = glob.glob(images_dir + "/*.jpg")
    mask_files  = glob.glob(masks_dir + "/*.jpg")
    
    print("--- num image_files {}".format(len(image_files)))
    print("--- num mask_files {}".format(len(mask_files)))

    index = 10000
    for image_file in image_files:
      print("== image_file {}".format(image_file))
      basename     = os.path.basename(image_file)
      mask_file   =  os.path.join(masks_dir, basename)
      index += 1
      if os.path.exists(image_file) and os.path.exists(mask_file):
         image = Image.open(image_file).convert("RGB")
         square_image = self.resize_to_square(image)
         self.augment(square_image,  output_images_dir, basename)
         mask  = Image.open(mask_file).convert("RGB")

         square_mask = self.resize_to_square(mask)
         self.augment(square_mask,  output_masks_dir, basename)

      else:
        print("Not found segmentation file {} corresponding to {}".format(image_file, mask_file))


if __name__ == "__main__":
  try:
    images_dir        = "./Liver-base/images"
    masks_dir         = "./Liver-base/masks"
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
    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


