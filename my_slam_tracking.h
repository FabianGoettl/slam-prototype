#pragma once

#include <iostream>
#include <list>

#include <opencv2/opencv.hpp>

namespace slam {

	class FeatureTrack {
	public:
		cv::Point2f base_position;
		cv::Point2f active_position;
		cv::Mat1f descriptor;
		int missed_frames;
	};

};