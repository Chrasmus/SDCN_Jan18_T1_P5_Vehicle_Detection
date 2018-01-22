**Vehicle Detection Project**

## Writeup

### Udacity Course, October 2017 cohort

**Self-Driving Car Engineer Nanodegree Program**

**Project 'Advanced Lane Lines', January 2018**

**Claus H. Rasmussen**

---

**Detect and track vehicles using color and gradient features and a support vector machine classifier**

---

**Vehicle Detection and Tracking**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_image]: ./output_images/car_image0152.png "Car image no. 0152 from the GTI_right dataset"
[notcar_image]: ./output_images/notcar_extra26.png "Not_car image from the Extras dataset"
[car-and_hog_9_8_2]: ./output_images/car-and_hog_9_8_2.png
[sliding_windows]: ./output_images/sliding_windows.png
[test_image1_w_detections]: ./output_images/test_image1_w_detections.png
[video_frame_20s_w_detections]: ./output_images/video_frame_20s_w_detections.png
[video_frame_21s_w_detections]: ./output_images/video_frame_21s_w_detections.png
[video_frame_21s_w_heatmap_no_thresh]: ./output_images/video_frame_21s_w_heatmap_no_thresh.png
[video_frame_21s_w_heatmap_thresh_7]: ./output_images/video_frame_21s_w_heatmap_thresh_7.png
[video1]: ./output_images/project_video_w_detections.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it - this is README.md writeup.

The code referenced here can be found in the Jupyter Notebook `Vehicle_detection.ipynb`. It contains some test code in some of the cells.
The notebook is provided as HTML in the file `Vehicle_detection.html`.

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car
![alt text][car_image]
Not car
![alt text][notcar_image]

I then explored different color spaces (code cell no. 3) and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][car-and_hog_9_8_2]

The code for extracting HOG features is contained in code cell no. 8 in the IPython notebook.  

---
#### 2. Explain how you settled on your final choice of HOG parameters.

I tried some combinations of parameters and ended up using the following:

| Parameter     | Value         |
|:-------------:|:-------------:|
|orient         | 9             |
|pix_per_cell   | 8             |
|cell_per_block | 2             |
|color space    | YCrCb         |
|:-------------:|:-------------:|

The color space was the most difficult parameter to establish. For a long time I had settled for LUC, but ongoing difficulties with detecting the white car in the frames around 19 to 23 seconds, eventually made me try YCrCb, which helped a little.

---
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM machine learning model using a combination of spatial features, color histograms and HOG features.
The spatial features and color histograms are computed in code cells no. 5 and 7.
The actual training code can be found in code cell no. 18, with a test accuracy equal to 0.989 (which is scary high and perhaps indicating overfitting).

---
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched with three scales (64x64, 96x96, and 128x128) in the lower part of the image, shown here below:

![alt text][sliding_windows]

Code is in code cell no. 21 (*function slide_window(...)*) and the result, the boxes, is fed into *function search_window(...)* (code cell no. 26).


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the pipeline I searched on three scales (1, 1.5, 2) using spatially binned color and histograms of color plus YCrCb 3-channel HOG features in the feature vector, which provided a nice result. Code is in code cell no. 42 (*function process_one_image(...)*)  Here are some example images:

Test_image1 with detections
![alt text][test_image1_w_detections]

Video frame at 20 sec. with one detection
![alt text][video_frame_20s_w_detections]

Most of my time optimizing, was dealing with the colorspace to be used. After trying the RGB and LUC, I settled with YCrCb, which gave me the best result.
Second I spend quite some time implementing a queue to provide a provide a 5 frame average in order to stabilize the wobling detection boxes. The implementation is found at the end of the function *process_one_image(...)*, in code cell no. 42.

---
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The pipeline is based on the function `process_one_image(...)` in code cell no. 42.

Here's a [link to my video result](./output_images/project_video_w_detections.mp4)

Most of the false positives appears in areas with shadows, which I think point in the direction of the HOG features.
The white car misses detection when passing from a light area to a dark area in the video at about 25 secs. A little later, when the black cars passes by, the white car is detected again, seemingly without any trouble. This 'white car problem' has been the largest time sink in my project, and I have now decided to submit anyway. I don't seem to get any closer to a solution for this particular problem, I'm afraid. I have been looking through the images, that I've used, and I think there is a bias towards more darker cars than white cars - but this may just be me searching for an explanation to the 'white car problem'.
The time used pr frame is a problem - at one frame pr 5.85 sec it is far to slow for anything in real life. Right know I think it has been a result of all the technical debt, I've accumulated in this project trying to make perform better on detection. It might need a proper rewrite, or at least some 


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I decided through testing, that the threshold should be 7, as this was the value that filtered away most false positives and kept most cars. I added a queue, length=5, to hold the heatmaps. Using the average of the five lastest heatmaps, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  Assuming each blob corresponded to a vehicle,  I constructed bounding boxes to cover the area of each blob detected. The code to handle the heatmaps are located in code cell no. 28 and 29. Below are images of heatmaps, without and with threshold=7, and the frame at 21 secs with resulting bounding box.

Heatmap, no thresholding:
![alt text][video_frame_21s_w_heatmap_no_thresh]

Heatmap, thresholding= 7:
![alt text][video_frame_21s_w_heatmap_thresh_7]

Image with resulting bounding box:
![alt text][video_frame_21s_w_detections]


---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project has been exiting in the way, that I've learned to process an image/video for features. I learned a lot, but I also find that this last project was the hardest problem to solve. It did take some time, before I understood, what the different parts of the code did, but when I completed the pipeline in the *process_one_mage(...)* function, I finally saw the light and was able to go hunting for where to tune the model and the code. I see a great potential with this pipeline to research more into some metrics for using different combinations of parameters and focusing on the difficult frames in the video. Also the two advanced videos from the previous project would be interesting to investigate further.

If I were going to pursue this project further, I would start with getting a better understanding of the color spaces and where to use them. Second, I would like to be a lot better a handling HOG features, both in term of what a car look like, but also cyclist and pedestrians. And traffic signs and other roadside equipment. Finally I think this project could evolve in using multiple ML models (using different color spaces etc.) and put a ML model on top of their results - a sort of a cascading model. But for now I think, I just have to work a little harder on the upcoming projects in Term 2 and 3 :-)

Kind regards, Claus H. Rasmussen
