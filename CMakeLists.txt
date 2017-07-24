cmake_minimum_required(VERSION 2.8.11)
project(stereo_match)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc imgcodecs highgui )
include_directories($(OpenCV_INCLUDE_DIRS))

find_package(PCL 1.8.0 REQUIRED COMPONENTS common io registration segmentation visualization)
include_directories(${PCL_INCLUDE_DIRS})


add_executable(stereo_match stereo_match.cpp)
target_link_libraries(stereo_match ${OpenCV_LIBS} ${PCL_LIBRARIES})
