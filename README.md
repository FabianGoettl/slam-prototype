# SLAM prototype
Prototype to implement a SLAM algorithm based on OpenCV and PCL functions. For each frame we extract keypoints, do a point triangulation and create a PCL point cloud.
The position of the camera is estimated by registration of recent and last point cloud.

The project's development was halted, because in this approach estimation errors are summed up over time. Hence, the camera pose drifts. Further filtering would be required to compensate errors.

