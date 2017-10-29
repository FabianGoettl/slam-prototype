# Stereo SLAM based on OpenCV and PCL
This is a C++ Stereo SLAM prototype implementation based on OpenCV and PCL.
The system requires two stereo calibrated USB webcams.

## Approach
In each frame we extract ORB features and match them with features of previous frames.
If matches are found along the epipolar line, we triangulate the points and determine their 3D position.

Based on the 3D-3D correspondences, we estimate the transformation between the frames by SVD decomposition with RANSAC.

Finally, the active points are transformed and merged into a PCL point cloud.

## Calibration
We use the [OpenCV Stereo calibration routine](https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_calib.cpp) that can be found in the official OpenCV example directory.

## Issues
The implementation does not include global optimization.
Estimation errors are not reduced and will be summed up over time.
In order to compensate errors, further investigations in pose graph optimization (such as implemented in the g2o library) would be required.