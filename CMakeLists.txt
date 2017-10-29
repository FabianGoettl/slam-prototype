cmake_minimum_required(VERSION 2.8.11)
project(slamPrototype)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc imgcodecs highgui )
include_directories($(OpenCV_INCLUDE_DIRS))

find_package(PCL 1.8.0 REQUIRED COMPONENTS common io registration segmentation visualization)
include_directories(${PCL_INCLUDE_DIRS})


set(SOURCES my_slam_prototype.cpp my_slam_tracking.cpp my_slam_tracking.h)

add_executable(slamPrototype ${SOURCES} )
target_link_libraries(slamPrototype ${OpenCV_LIBS} ${PCL_LIBRARIES})
