**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car_not_car_example.png
[image2]: ./writeup_images/HOG_example.jpg
[image3]: ./writeup_images/test1.jpg
[image4]: ./writeup_images/test_images.PNG
[image5]: ./writeup_images/six_frames.PNG
[image6]: ./writeup_images/final_frame_heatmap.PNG
[image7]: ./writeup_images/final_frame_bbox.PNG
[video1]: ./project_video.mp4
[video2]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook `CarND-Vehicle-Detection.ipynb`
The function name is `get_hog_features`. It uses `skimage.feature.hog` to generate the hog feature.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle`
 and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and
 `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what 
 the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and 
`cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I have run a grid search on the HOG parameters and others (over 3000 combinations) to search for the best parameter. 
According to the result, I select the following set of parameter (`color space = YCrCb`, `orientations = 9`, 
`pixels_per_cell = 16`, and  `cells_per_block = 2`), which has a test accuracy score of 99.87%
 
Here is the list of value I used to run grid search

| Parameter        | Values   | 
|:-------------:|:-------------:| 
| Color space     | RGB, HSV, LUV, YCrCb    | 
| Orientation      | 5, 9      |
| Pixel per cell      | 8, 16     |
| Cell per block      | 2, 3, 4      |
| Hog channel     | 0, 1, 2, ALL   |
| Spatial size     | 16,32   |
| Histogram bins     | 16,32   |
| Spatial feature included     | True, False   |
| Histogram feature included     | True, False  |
| Hog feature included     | True   |

I first train over 3000s linear SVM (use only 2000 images) and check for their test accuracy score, 
and then I select the best 650 hyper-parameter sets and train a linear SVM using the full set of training data.

Here are the best 5 hyper-parameter set I can find.

| Parameter        | 1   |  2   |  3   |  4   |  5   | 
|:-------------:|:-------:| :-------:|:-------:|:-------:|:-------:|
| Color space     | YCrCb    | LUV    |LUV    |YCrCb    |HSV    |
| Orientation      | 9      | 9      | 9      | 9      | 9      |
| Pixel per cell      | 16      |16      |16      |16      |16      |
| Cell per block      | 2  | 3  | 3  | 2  | 3  |
| Hog channel     | ALL   |ALL   |ALL   |ALL   |ALL   |
| Spatial size     | 32   |32   |32   |16   |16   |
| Histogram bins     | 32   |32   |32   |16   |32   |
| Spatial feature included     | True   | True   | False   | True   | True   |
| Histogram feature included     | True  | True  | True  | True  | True  |
| Hog feature included     | True   | True   | True   | True   | True   |
| Accuracy     | 99.87%   | 99.87%   | 99.87%   | 99.83%   | 99.83%   |

If you are interested, you can take a look at 2 Excel files: `first_round_statistics.xlsx` and `second_round_statistics.xlsx`
 in folder `hyper_param_grid_search`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 5th, 6th and 7th code cell of the IPython notebook 
`CarND-Vehicle-Detection.ipynb`. Under the title "Train a Linear SVC for classification".

I trained a linear SVM using the training dataset provided by the course website. Here is the 
[link of small dataset](vehicle_small_dataset.7z), and [link of full dataset](vehicle_dataset.7z).

I also include 2 more feature sets beside the HOG features.

* The spatial feature: I first scale the image to `spatial_size` and then 
flatten the image to create an array of number. For example, if I use `spatial_size = 32 x 32`, I first scale the image
to the size `32 x 32` and use each of the pixels as an input feature, which is `1024` in total. The implementation is 
the function `bin_spatial` in the 2nd code cell of the IPython notebook `CarND-Vehicle-Detection.ipynb` 

* The color histogram, the image is seperated into 3 channels and calculate the histogram respectively, the bin size of the histogram
is specified in `hist_bins`. These 3 histograms are then combined together to create a single histogram, which gives 
`3 X hist_bin features`. In the case when histogram bin number is 32, the resulting histogram is of length `3 X 32 = 96`.
The implementation is the function `color_hist` in the 2nd code cell of the IPython notebook 
`CarND-Vehicle-Detection.ipynb` 

To start training a classifier, first I convert all the image in the training set from the original format *.png to *.jpeg. 
It is because the testing images and testing videos are all in the form of *.jpeg. The numeric value read from *.png 
and that from *.jpeg are of different range.

Second, I vertically flip all the image, such that the right side of the original image becomes the left side of the 
resulting image. so that the number of training dataset is double.

After that, I shuffle the data set and split it into training and test dataset.

Then, I fit the training dataset to a linear SVC.

Finally, I calculate the accuracy score and make sure that the test score is higher than my minimum requirement.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The function `find_car` in the 8th code block of the IPython notebook 
`CarND-Vehicle-Detection.ipynb` implemented a sliding window search. This function takes the following parameters:
* img: Image
* color_space: color space used in extracting features
* ystart, ystop: the start and the end of y pos for searching cars
* scale: scale factor. passing value > 1 will make the image for searching smaller.
* svc: Support vector machine
* X_scaler: scaler to standardize features
* orient, pix_per_cell, cell_per_block: HOG parameters
* spatial_size: specify the size of spatial feature
* hist_bins: specify the size of the histogram feature
* hog_channel: specify which channel hog feature is extracted
* spatial feat, hist_feat, hog_feat: enable / disable features

As the function `find_car` uses a pre-calculated the hog feature, and hence the step is a multiple of pix_per_cell, the smaller the pix_per_cell,
more windows this function can search.

For pix_per_cell = 16 and scale = 1, the sliding windows search for following windows. each unit square in the below image
 is having sides of length `pix_per_cell X scale = 16 X 1 = 16 pixels`, each window has a side of `64 pixels` long. which is
 specified in the variable `window`.

The function also specified the step size by cells_per_step. In this project, each step contains 1 cell `(16 pixels)`.
If I move one step along the x-axis, the overlapping area is `(64 - 16) X 64 = 48 X 64 = 3072 pixels`.

Here is a Python code segment
```python 
# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
window = 64
nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
cells_per_step =1  # Instead of overlap, define how many cells to step
nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
```

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 1 scale `(scale = 1)` using YCrCb 3-channel HOG features plus spatially binned color 
and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

From the 6 images, only 1 car in an image is missed, and no false positive.

In the overall, True Positive = 9, False Positive = 0, False Negative = 1.

![alt text][image4]

To improve the performance, I only search for the bottom of the image, from 400th pixels to 656th pixels along the y-axis.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created 
a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` 
to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed 
bounding boxes to cover the area of each blob detected.

The bounding boxes in the previous frame is kept for current frame processing, but the weighting is exponentially decay
at a rate of 0.5, all bounding boxes have weighting less than 0.1 will be purged.

Here's an example result showing the heatmap from a series of frames of video, the result of 
`scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of the video:

##### Here are six frames and their corresponding heatmaps:

![alt text][image5]

##### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

##### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail 
and how I might improve it if I were going to pursue this project further.  

In this project, I have used HOG feature and SVM with an additional feature to do a sliding window search to find cars
in the video. 

##### HOG-SVM

1. HOG-SVM performs well in this project. Even if the feature set has a very high dimension. In this case, the dimension
of the feature set is of `4140`, and the number of the training sample is only `5966 + 8968 = 14935`.
I was afraid SVM overfits the training data and produce a poor output, but it turns out this is not the case.

2. Hyper-parameter tuning is very time-consuming. Given that I have to find not only C (SVM parameters), but also
HOG parameter which are `(orientations, pixels_per_cell and cells_per_block)`. I do not have time for this project
to implement the hyper-parameter grid search efficiently, but by brute force find out the hyper-parameters. (by inspecting 
over 3000 combinations.) 

3. Instead of using HOG-SVM, Decision tree can also be used in the detection process. 
In sklearn, there is a VotingClassifier[1] to combine multiple classifiers into a single classifier to improve the
prediction accuracy.

##### Sliding windows search:

1. In this project, I used only 1 scale. The sliding windows search can be improved by scaling the image multiple time, 
such that objects of different size can be detected without missing. Missing objects (True negative) can lead to 
traffic accidents. The reason I only use 1 scale is because using more scale can further reduces the frame processing 
rate < 1 frame/second.

2. I tried to use Python multiprocessing library to improve the runtime. it is because each window is independent of each
other when finding if there is a car within the windows or not. But I find passing/sharing numpy NDArray
among Python process difficult so I gave up this idea. If this idea can be implemented, it should increase the runtime >
1.4 frames/ second. 

##### Object tracking:
1. In this project, there is no object tracking. The pipeline only outputs what it thinks which part of the image contains
cars, but it does not know is this object was in the previous frames before. 
To perform object tracking, we can use the color histogram[2]. Color histogram should be able to matches objects
between frames. Another advance technique like SIFT[3] can be applied to track object too.

##### Integration with previous project:
1. I have also integrated the previous project "Advance lane finding" to this project such that the program can now
detect lane and vehicle `vehicle_detection.py`. By doing so, the computer now has more information on the road and
decides what is the next action (accelerate, decelerate, emergency stop) and route planning.
Here is [a link to the video](https://youtu.be/rEKE_fyPhqY) The program takes 2 parameters, the video file name and a 
debug flag. To run the program, 
```bash
python vehicle_detection.py project_video.mp4 false
```
---
### Reference

[1] http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

[2] Pérez, P., Hue, C., Vermaak, J., & Gangnet, M. (2002). Color-based probabilistic tracking. Computer vision—ECCV 2002, 661-675.

[3] Zhou, H., Yuan, Y., & Shi, C. (2009). Object tracking using SIFT features and mean shift. Computer vision and image understanding, 113(3), 345-352.

