# Image-Segmentation-Liver-Tumor (2023/08/08)
<h2>
1 Image-Segmentation-Liver-Tumor 
</h2>
<p>
This is an experimental project for Image-Segmentation of Liver-Tumor by using
 <a href="https://github.com/atlan-antillia/Tensorflow-Slightly-Flexible-UNet">Tensorflow-Slightly-Flexible-UNet</a> Model,
which is a typical classic Tensorflow2 UNet implementation <a href="./TensorflowUNet.py">TensorflowUNet.py</a> 
<p>
The image dataset used here has been taken from the following web site.
</p>

<pre>
3D Liver segmentation
https://www.kaggle.com/datasets/prathamgrover/3d-liver-segmentation
</pre>
<b>About Dataset</b>
<pre>
Use this dataset for segmenting the liver tumor in 3D scans. The imagesTr files contains nifti images 
which are input for this image. Each nifti image contains multiple 2D slices of a single scan.
labelsTr contains the output for the corresponding input specifying where the tumor is localised.
</pre>
<b>License</b><br>
<a href="https://opendatacommons.org/licenses/dbcl/1-0/">
Open Data Commons
</a>

<br>
<br>
<h2>
2. Install Image-Segmentation-Liver-Tumor 
</h2>
Please clone Image-Segmentation-Liver-Tumor.git in a folder <b>c:\google</b>.<br>
<pre>
>git clone https://github.com/sarah-antillia/Image-Segmentation-Liver-Tumor.git<br>
</pre>
You can see the following folder structure in your working folder.<br>

<pre>
Image-Segmentation-Liver-Tumor 
├─asset
└─projects
    └─-Liver-Tumor
        ├─eval
        ├─generator
        ├─mini_test
        ├─models
        ├─Liver-Tumor
        │   ├─test
        │   │  ├─images
        │   │  └─masks
        │   ├─train
        │   │  ├─images
        │   │  └─masks
        │   └─valid
        │       ├─images
        │       └─masks
        ├─test_output
        └─test_output_merged
</pre>

<h2>
3 Prepare dataset
</h2>

<h3>
3.1 Download master dataset
</h3>
  Please download the original image and mask dataset <b>3D Liver segmentation</b> from the following link<br>

<pre>
3D Liver segmentation
https://www.kaggle.com/datasets/prathamgrover/3d-liver-segmentation
</pre>
</pre>
The dataset <b>3D Liver segmentation</b> has the following folder structure.<br>
<pre>
./Task03_Liver_rs
├─imagesTr
└─labelsTr
</pre>
These folders contain 300 liver_*.nii files repectively.<br>
<h3>
3.2 Create base image and mask dataset
</h3>
Please run Python script <a href="./projects/Liver-Tumor/generator/create_base.py">create_base.py</a>.
to create jpg image and mask files.<br>
<pre>
>python create_base.py
</pre>
, by which Liver-base dataset will be created.<br>
<pre>
./Liver-base
├─images
└─masks
</pre>


<h3>
3.3 Create image and mask master dataset
</h3>

By using Python script <a href="./projects/Liver-Tumor/generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a>,
 we have created <b>Liver-Tumor-master</b> dataset from the <b>Liver-Base</b> dataset.<br>
The script performs the following image processings.<br>
<pre>
1 Create 256x256 square images from original jpg files in <b>Tumor-base/images</b> folder..
2 Create 256x256 square mask  corresponding to the Tumor-base images files. 
3 Create flipped and mirrored images and masks of size 256x256 to augment the resized square images and masks.
</pre>

The created <b>Liver-master</b> dataset has the following folder structure.<br>
<pre>
./Liver-master
 ├─images
 └─masks
</pre>

<h3>
3.4 Split master to test, train and valid 
</h3>
By using Python script <a href="./projects/Liver-Tumor/generator/split_master.py">split_master.py</a>,
 we have finally created <b>Liver-Tumor</b> dataset from the <b>Liver-Tumor-master</b>.<br>
<pre>
./Liver-Tumor
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>

<b>train/images samples:</b><br>
<img src="./asset/train_images_samples.png" width="1024" height="auto">
<br>
<b>train/masks samples:</b><br>
<img src="./asset/train_masks_samples.png"  width="1024" height="auto">
<br>
<br>
<b>Dataset inspection</b><br>
<img src="./asset/dataset_inspection.png" width="720" height="auto">
<br>
<h2>
4 Train TensorflowUNet Model
</h2>
 We have trained Liver-Tumor TensorflowUNet Model by using the following
 <b>train_eval_infer.config</b> file. <br>
Please move to ./projects/Liver-Tumor directory, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
, where train_eval_infer.config is the following.
<pre>
; train_eval_infer.config
; Dataset of Liver-Tumor
; 2023/08/08 (C) antillia.com

[model]
image_width    = 256
image_height   = 256
image_channels = 3
num_classes    = 1
base_filters   = 16
base_kernels   = (5,5)
num_layers     = 7
dropout_rate   = 0.07
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (2,2)
loss           = "bce_iou_loss"
;metrics        = ["iou_coef", "sensitivity", "specificity"]
metrics        = ["iou_coef"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
patience      = 10
metrics       = ["iou_coef", "val_iou_coef"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "./Liver-Tumor/train/images"
mask_datapath  = "./Liver-Tumor/train/masks"
create_backup  = True

[eval]
image_datapath = "./Liver-Tumor/valid/images"
mask_datapath  = "./Liver-Tumor/valid/masks"
output_dir     = "./eval_output"

[infer] 
;images_dir = "./mini_test/"
images_dir = "./Liver-Tumor/test/images"
output_dir = "./test_output"
merged_dir = "./test_output_merged"

[mask]
blur      = True
binarize  = True
threshold = 74
</pre>

The training process has just been stopped at epoch 43 by an early-stopping callback as shown below.<br><br>
<img src="./asset/train_console_output_at_epoch_43_0807.png" width="720" height="auto"><br>
<br>
<br>
<b>Train metrics line graph</b>:<br>
<img src="./asset/train_metrics.png" width="720" height="auto"><br>
<br>
<b>Train losses line graph</b>:<br>
<img src="./asset/train_losses.png" width="720" height="auto"><br>


<h2>
5 Evaluation
</h2>
 We have evaluated prediction accuracy of our Pretrained Liver-Tumor Model by using <b>valid</b> dataset.<br>
Please move to ./projects/Liver-Tumor/ directory, and run the following bat file.<br>
<pre>
>2.evalute.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>
The evaluation result is the following.<br>
<img src="./asset/evaluate_console_output_at_epoch_43_0807.png" width="720" height="auto"><br>
<br>

<h2>
6 Inference 
</h2>
We have also tried to infer the segmented region for 
<pre>
images_dir    = "./Liver-Tumor/test/images" 
</pre> dataset defined in <b>train_eval_infer.config</b>,
 by using our Pretrained Liver-Tumor UNet Model.<br>
Please move to ./projects/Liver-Tumor/ directory, and run the following bat file.<br>
<pre>
>3.infer.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../TensorflowUNetInferencer.py ./train_eval_infer.config
</pre>

<b><a href="./projects/Liver-Tumor/Liver-Tumor/test/images">Test input images</a> </b><br>
<img src="./asset/test_image.png" width="1024" height="auto"><br>
<br>
<b><a href="./projects/Liver-Tumor/Liver-Tumor/test/masks">Test input ground truth mask</a> </b><br>
<img src="./asset/test_ground_truth.png" width="1024" height="auto"><br>
<br>

<b><a href="./projects/Liver-Tumor/test_output/">Inferred images </a>test output</b><br>
<img src="./asset/test_output.png" width="1024" height="auto"><br>
<br>
<br>


<b><a href="./projects/Liver-Tumor/test_output_merged">Inferred merged images (blended test/images with 
inferred images)</a></b><br>
<img src="./asset/test_output_merged.png" width="1024" height="auto"><br><br>

<b>Some enlarged input images and inferred merged images</b><br>
<table>
<tr><td>Input:flipped_10020_125.jpg</td><td>Inferred merged:flipped_10020_125.jpg</td></tr>
<tr>
<td><img src = "./projects/Liver-Tumor/Liver-Tumor/test/images/flipped_10020_125.jpg" width="512" height="auto"></td>
<td><img src = "./projects/Liver-Tumor/test_output_merged/flipped_10020_125.jpg"  width="512" height="auto"></td>
</tr>

<tr><td>Input:flipped_10072_136.jpg</td><td>Inferred merged:flipped_10072_136.jpg</td></tr>
<tr>
<td><img src = "./projects/Liver-Tumor/Liver-Tumor/test/images/flipped_10072_136.jpg" width="512" height="auto"></td>
<td><img src = "./projects/Liver-Tumor/test_output_merged/flipped_10072_136.jpg"  width="512" height="auto"></td>
</tr>

<tr><td>Input:flipped_10072_142.jpg</td><td>Inferred merged:flipped_10072_142.jpg</td></tr>
<tr>
<td><img src = "./projects/Liver-Tumor/Liver-Tumor/test/images/flipped_10072_142.jpg" width="512" height="auto"></td>
<td><img src = "./projects/Liver-Tumor/test_output_merged/flipped_10072_142.jpg"  width="512" height="auto"></td>
</tr>

<tr><td>Input:flipped_10118_132.jpg</td><td>Inferred merged:flipped_10118_132.jpg</td></tr>
<tr>
<td><img src = "./projects/Liver-Tumor/Liver-Tumor/test/images/flipped_10118_132.jpg" width="512" height="auto"></td>
<td><img src = "./projects/Liver-Tumor/test_output_merged/flipped_10118_132.jpg"  width="512" height="auto"></td>
</tr>
<tr><td>Input:mirrored_10051_103.jpg</td><td>Inferred merged:mirrored_10051_103.jpg</td><tr>
<tr>
<td><img src = "./projects/Liver-Tumor/Liver-Tumor/test/images/mirrored_10051_103.jpg" width="512" height="auto"></td>
<td><img src = "./projects/Liver-Tumor/test_output_merged/mirrored_10051_103.jpg"  width="512" height="auto"></td>
</tr>
<tr><td>Input:mirrored_10121_83.jpg</td><td>Inferred merged:mirrored_10121_83.jpg</td><tr>
<tr>
<td><img src = "./projects/Liver-Tumor/Liver-Tumor/test/images/mirrored_10121_83.jpg" width="512" height="auto"></td>
<td><img src = "./projects/Liver-Tumor/test_output_merged/mirrored_10121_83.jpg"  width="512" height="auto"></td>
</tr>
<tr><td>Input:mirrored_10123_92.jpg</td><td>Inferred merged:mirrored_10123_92.jpg</td><tr>

<tr>
<td><img src = "./projects/Liver-Tumor/Liver-Tumor/test/images/mirrored_10123_92.jpg" width="512" height="auto"></td>
<td><img src = "./projects/Liver-Tumor/test_output_merged/mirrored_10123_92.jpg"  width="512" height="auto"></td>
</tr>
</table>

<br>
<h3>
References
</h3>
<b>1. 3D Liver segmentation</b><br>
<pre>
https://www.kaggle.com/datasets/prathamgrover/3d-liver-segmentation/code
</pre>

<b>2. 3D liver unet</b><br>
<pre>
https://www.kaggle.com/code/prathamgrover/3d-liver-unet
</pre>

<b>3. Liver segmentation 3D-IRCADb-01</b><br>
<pre>
https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/
</pre>
