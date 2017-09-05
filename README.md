# SLAM prototype
This is a prototype to implement a SLAM algorithm based on OpenCV and PCL functions. For each frame we extract keypoints, do a point triangulation and create a PCL point cloud.
The position of the camera is estimated by registering the recent and last point cloud.

The project's development was halted, because estimation errors are summed up over time. Hence, the camera pose drifts. In order to compensate errors, further filtering would be required e.g. by a Kalman filter.

