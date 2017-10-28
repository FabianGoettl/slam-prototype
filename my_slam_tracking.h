#pragma once

#include <iostream>
#include <list>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace slam {

	class FeatureTrack {
	public:
		cv::Point2f base_position;
		cv::Point2f active_position;

		cv::Point3f base_position_3d;
		cv::Point3f active_position_3d;

		cv::Mat descriptor;
		int missed_frames;
	};

	class TrackedView {
	public:
		cv::Matx33f R;
		cv::Matx31f T;
	};

	class TrackingSharedData {
	public:
		//Commands
		bool is_data_new;   // True if this class has new data that should be rendered
		bool is_tracking_enabled; //True if the tracking thread should process images

		cv::Matx33f base_R;
		cv::Matx31f base_T;
		cv::Mat3b *base_rgb;
		//cv::Mat3f *base_pointmap;

		//Last tracked image
		cv::Mat3b *active_rgb;
		//cv::Mat1s *active_depth;

		//Model
		std::list<FeatureTrack> tracks; // Tracked features since last base frame
		std::vector<TrackedView> views; // All registered views

		TrackingSharedData();
		~TrackingSharedData();
	};

	class TrackingModule
	{
	public:
		//void compute_pointmap(const cv::Mat1s &depth, cv::Mat3f &pointmap);
		//void cloud_from_pointmap(const cv::Mat3b &rgb, const cv::Mat3f &pointmap, boost::container::vector<pcl::PointXYZRGB> &cloud);

		static void match_features(const std::list<FeatureTrack> &tracks, const cv::Mat &new_descriptors, std::vector<int> &match_idx);

		static void update_tracks(std::list<FeatureTrack> &tracks, const std::vector<cv::KeyPoint> &feature_points, const cv::Mat &feature_descriptors, 
			std::vector<int> &match_idx, std::vector<cv::Point3f>& pnts3D);

		static bool is_track_stale(const FeatureTrack &track);
		
		static float get_median_feature_movement(const std::list<FeatureTrack> &tracks);

		static void get_tracked_keypoints(const std::vector<cv::KeyPoint> &feature_points, std::vector<cv::KeyPoint> &feature_output, const std::vector<int> &match_idx) {
			
			for (auto it = match_idx.begin(); it != match_idx.end(); ++it) {
				int idx = *it;
				if(idx > 0)
					feature_output.push_back(feature_points.at(idx));
			}
		}

		static void triangulate_matches(std::vector<cv::DMatch>& matches, const std::vector<cv::KeyPoint>&keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
			cv::Mat& cam1P, cv::Mat& cam2P, std::vector<cv::Point3f>& pnts3D);

		static void transformation_from_tracks(const std::list<FeatureTrack> &tracks, cv::Matx33f &R, cv::Matx31f &T);
		
		static void ransac_transformation(const cv::Mat1f &X, const cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T);
		static void absolute_orientation(cv::Mat1f &X, cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T);
	};

}