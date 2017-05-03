## Advanced Lane Finding Project for Udacity Self-Driving Car Nanodegree

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard images using the `cv2.undistort()` function and obtained this result: 

![alt text][1]

  [1]: ./test_images/Original+Undistort2.png "Original and Undistorted Chessboard"

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][2]

  [2]: ./test_images/test_undist_road.png "Original Test Image"
  
Here is what it looked like after it is undistorted. It is not very noticeable without well defined lines like the chessboard had. However, this step is very important it creating an accurate mapping of the image points to the realworld. This helps to maintain accurate radii of curvature and distance measurements.
  
![alt text][2]

  [2]: ./test_images/Original+Undistort.png "Original and Undistorted Road Image"

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for creating a thresholded binary is in code cell number 6 in the IPython notebook. 
This task is really about trying to remove as much data that we can from the image and retain only pixels that represent the lane lines. I applied a few thresholding techniques learned in the lessons including: Color space transformation, gradient magnitude, gradient direction and color thresholding. Below are a few examples of the thresholding with binary rgb channels. I later change the rgb binary into a 'white hot' binary with only one channel. This helped with pipeline processing.

![alt text][2]

  [2]: ./test_images/ "test"
  
![alt text][2]

  [2]: ./test_images/ "test"

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `line_warp()`, which appears in the 8th code cell of the IPython notebook.  The `lines_warp()` function takes as inputs an image (`img`), as well as `inv=0` (`inv=1` performs inverse transform) and `draw=None` (`draw=True` draws the perspective line for visualization)  I chose the hardcode the source and destination points in the following manner:

```python
imshape = color_grad.shape
# White and yellow video Values
lbxper = .14 # left bottom x vertice percentage
rbxper = .89 # right bottom x vertice percentage
byper = 1 # bottom y vertices percentage
typer = .63 # top y vertices percentage
lm = -0.453 # left slope 
rm = 0.6 # right slope

# Vertices calculations
xlb = imshape[1]*lbxper
ylb = imshape[0]*byper
xlt = (imshape[0]*typer - ylb)/lm
ylt = imshape[0]*typer
xrb = imshape[1]*rbxper
yrb = ylb
xrt = (ylt - ylb)/rm + xrb
yrt = ylt

vertices = np.array([[(xlb,ylb),(xlt, ylt), (xrt, yrt), (xrb,yrb)]], dtype=np.int32)

# Define polygon vertices for masking
left_bottom = vertices[0][0]
right_bottom = vertices[0][3]
left_top = vertices[0][1]
right_top = vertices[0][2]

if draw:
    # Draw perspective lines on img
    cv2.line(img, (left_bottom[0], left_bottom[1]), (left_top[0], left_top[1]), (255,0,0), 1)
    cv2.line(img, (right_bottom[0], right_bottom[1]), (right_top[0], right_top[1]), (255,0,0), 1)
    cv2.line(img, (left_bottom[0], left_bottom[1]), (right_bottom[0], right_bottom[1]), (255,0,0), 1)
    cv2.line(img, (left_top[0], left_top[1]), (right_top[0], right_top[1]), (255,0,0), 1)


# Perpective warp

offset = 250
# define 4 source points src = np.float32([[,],[,],[,],[,]])
src = np.float32([[left_bottom[0], left_bottom[1]], [right_bottom[0], right_bottom[1]], 
                 [right_top[0], right_top[1]], [left_top[0], left_top[1]]])
#     print('Source Corners = ', src)

# define 4 destination points dst = np.float32([[,],[,],[,],[,]])
dst = np.float32([[offset, imshape[0]], [imshape[1]-offset, imshape[0]], 
                                 [imshape[1]-offset, 0], 
                                 [offset, 0]])
```

This resulted in the following source and destination points:
Source Corners =  
[[  179., 720.]
 [ 1139., 720.]
 [  695., 453.]
 [  588., 453.]]
Destination Corners =  
[[  250., 720.]
 [ 1030., 720.]
 [ 1030., 0.]
 [  250., 0.]]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. You can see examples of the warped images in normal vision and binary vision. In all cases the `src` and `dst` lines are parrallel in the warped image. In the last image I show that the curved lines are also parallel.

![alt text][4]

  [4]: ./test_images/Warped-Binary-Curved.png "Warped Original"
  
![alt text][5]

  [5]: ./test_images/Warped-Binary-Curved1.png "Warped Binary Straight"
  
![alt text][6]

  [6]: ./test_images/Warped-Binary-Curved2.png "Warped Binary Curved"

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels and polynomial fitting were done with the function `find_lines()` in code cell 24 of the IPython notebook. The warped binary thresholed image is given as input. Then a histogram is calculated of the binary to see where the lines are most likely to be. Then a sliding window search is done from the bottom of the image to the top in `n` increments with a window width of 75 pixels. Anytime a pixel falls inside the window and there are at least 50 of them it considers it a line and stores the pixels for later fitting. Once all the detected line pixels are stored a polynomial of order 2 is fit to each the left and right lines.

After the first run of the sliding window search, a better method is used to search since we already know where the lines were from the previous frame of the video. So we start looking there with a margin around the lines. If we end up loosing the line we start over with the sliding window search.

![alt text][7]

  [7]: ./test_images/Sliding-Window-PolyFit.png "Sliding Window Fit"
  
![alt text][8]

  [8]: ./test_images/Skip-Sliding-Window-PolyFit.png "Previous Detection Fit"

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature of each lane line is done in the function `radius()` which takes as inputs y values of the image as well as the x values of the most current fit. The equation used for this is shown below.

Radius_of_curve =[(1+(2Ay+B)^2)^3/2]/∣2A∣

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][9]

  [9]: ./test_images/Drawlane1.png "Original"
  
![alt text][10]

  [10]: ./test_images/Drawlane2.png "Warped"
  
![alt text][11]

  [11]: ./test_images/Drawlane3.png "Lane Detection"
  
![alt text][12]

  [12]: ./test_images/Drawlane4.png "Painted Lane"

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
