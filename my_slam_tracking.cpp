#include "my_slam_tracking.h"

// Ratio to the second neighbor to consider a good match.
#define RATIO    0.6

// Max z-value (depth in mm)
// 40*stereobaseline is usually a good value
#define MAX_Z 3800

using namespace cv;
using namespace std;

namespace slam {

	TrackingSharedData::TrackingSharedData() :
		is_data_new(false),
		is_tracking_enabled(true),
		base_rgb(new cv::Mat3b(480, 640)),
		//base_pointmap(new cv::Mat3f(480, 640)),
		active_rgb(new cv::Mat3b(480, 640)),
		//active_depth(new cv::Mat1s(480, 640)),
		base_R(1, 0, 0, 0, 1, 0, 0, 0, 1),
		base_T(0, 0, 0)
	{
	}

	TrackingSharedData::~TrackingSharedData() {
		delete base_rgb;
		//delete base_pointmap;
		delete active_rgb;
		//delete active_depth;
	}

	/**
	Matches new ORB features with tracks
	*/
	void TrackingModule::match_features(const std::list<FeatureTrack> &tracks, const cv::Mat &new_descriptors, std::vector<int> &match_idx) {

		// Create a FlannMatcher based on values provided in docs
		FlannBasedMatcher matcher(new cv::flann::LshIndexParams(12, 20, 2));

		vector<Mat> train_vector;
		train_vector.push_back(new_descriptors);

		matcher.add(train_vector);

		match_idx.resize(tracks.size());

		int match_count = 0;
		list<slam::FeatureTrack>::const_iterator track_it;
		int i;

		for (i = 0, track_it = tracks.begin(); track_it != tracks.end(); i++, track_it++) {
			vector<vector<DMatch>> matches;
			matcher.knnMatch(track_it->descriptor, matches, 2);

			if (matches[0].size() >= 2) {
				// Do a second neighbor ratio test
				float best_dist = matches[0][0].distance;
				float next_dist = matches[0][1].distance;

				if (best_dist < next_dist * RATIO) {
					match_idx[i] = matches[0][0].trainIdx;
					match_count++;
				}
				else {
					match_idx[i] = -1;
				}
			}
		}

		cout << "Matched features:" << match_count << endl;
	}

	void TrackingModule::update_tracks(std::list<FeatureTrack> &tracks, const std::vector<cv::KeyPoint> &feature_points,
		const cv::Mat &feature_descriptors, std::vector<int> &match_idx, std::vector<cv::Point3f>& pnts3D) {

		std::list<FeatureTrack>::iterator track_it;
		int i, updated = 0, missed = 0;
		for (i = 0, track_it = tracks.begin(); track_it != tracks.end(); i++, track_it++) {
			int j = match_idx[i];
			if (j >= 0) {

				Point3d point = pnts3D[j];

				if (point.x != 0 && point.y != 0 && point.z != 0) {
					track_it->missed_frames = 0;
					track_it->active_position = feature_points[j].pt;
					track_it->active_position_3d = pnts3D[j];
					feature_descriptors.row(j).copyTo(track_it->descriptor);
					updated++;
					continue;
				}
			}
			// If not updated yet
			track_it->missed_frames++;
			missed++;
		}

		cout << "Updated Tracks:" << updated << " Missed:" << missed;

		// Delete tracks
		tracks.remove_if(TrackingModule::is_track_stale);
	}

	bool TrackingModule::is_track_stale(const FeatureTrack &track) {
		return track.missed_frames > 10;
	}

	float TrackingModule::get_median_feature_movement(const std::list<FeatureTrack> &tracks) {

		std::vector<float> vals;
		std::list<FeatureTrack>::const_iterator track_it;
		for (track_it = tracks.begin(); track_it != tracks.end(); track_it++) {

			if (track_it->missed_frames == 0) {
				// Diff. of base and active position
				vals.push_back(fabs(track_it->base_position.x - track_it->active_position.x) +
					fabs(track_it->base_position.y - track_it->active_position.y));
			}
		}

		if (vals.empty())
			return 0;
		else {
			int n = vals.size() / 2;

			// Sort vector
			std::nth_element(vals.begin(), vals.begin() + n, vals.end());

			return vals[n];
		}
	}

	void TrackingModule::transformation_from_tracks(const std::list<FeatureTrack> &tracks, cv::Matx33f &R, cv::Matx31f &T) {
		std::list<FeatureTrack>::const_iterator track_it;

		cv::Mat1f X(0, 3), Y(0, 3);
		X.reserve(tracks.size());
		Y.reserve(tracks.size());

		for (track_it = tracks.begin(); track_it != tracks.end(); track_it++) {

			if (track_it->missed_frames != 0)
				continue;

			const cv::Point3f &base_point = track_it->base_position_3d;
			if (base_point.z <= 0)
				continue;

			const cv::Point3f &active_point = track_it->active_position_3d;
			if (active_point.z <= 0)
				continue;

			// Add new row to matrices
			int i = X.rows;
			X.resize(i + 1);
			X(i, 0) = base_point.x;
			X(i, 1) = base_point.y;
			X(i, 2) = base_point.z;

			Y.resize(i + 1);
			Y(i, 0) = active_point.x;
			Y(i, 1) = active_point.y;
			Y(i, 2) = active_point.z;
		}

		if (X.rows > 0 && Y.rows)
			ransac_transformation(X, Y, R, T);
	}

	void TrackingModule::ransac_transformation(const cv::Mat1f &X, const cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T) {
		const int max_iterations = 200;
		const int min_support = 3;
		const float inlier_error_threshold = 0.2f * 1000;

		const int pcount = X.rows;
		cv::RNG rng;
		cv::Mat1f Xk(min_support, 3), Yk(min_support, 3);
		cv::Matx33f Rk;
		cv::Matx31f Tk;
		std::vector<int> best_inliers;

		for (int k = 0; k < max_iterations; k++) {

			// Select random points
			for (int i = 0; i < min_support; i++) {
				int idx = rng(pcount);
				Xk(i, 0) = X(idx, 0);
				Xk(i, 1) = X(idx, 1);
				Xk(i, 2) = X(idx, 2);
				Yk(i, 0) = Y(idx, 0);
				Yk(i, 1) = Y(idx, 1);
				Yk(i, 2) = Y(idx, 2);
			}

			// Get orientation
			absolute_orientation(Xk, Yk, Rk, Tk);

			// Get error
			std::vector<int> inliers;
			for (int i = 0; i < pcount; i++) {
				float a, b, c, errori;
				cv::Matx31f py, pyy;
				py(0) = Y(i, 0);
				py(1) = Y(i, 1);
				py(2) = Y(i, 2);
				pyy = Rk*py + T;
				a = pyy(0) - X(i, 0);
				b = pyy(1) - X(i, 1);
				c = pyy(2) - X(i, 2);
				errori = sqrt(a*a + b*b + c*c);
				if (errori < inlier_error_threshold) {
					inliers.push_back(i);
				}
			}

			if (inliers.size() > best_inliers.size()) {
				best_inliers = inliers;
			}
		}
		std::cout << "Inlier count: " << best_inliers.size() << "/" << pcount << endl;

		// Do final estimation with inliers
		Xk.resize(best_inliers.size());
		Yk.resize(best_inliers.size());

		for (unsigned int i = 0; i < best_inliers.size(); i++) {
			int idx = best_inliers[i];
			Xk(i, 0) = X(idx, 0);
			Xk(i, 1) = X(idx, 1);
			Xk(i, 2) = X(idx, 2);
			Yk(i, 0) = Y(idx, 0);
			Yk(i, 1) = Y(idx, 1);
			Yk(i, 2) = Y(idx, 2);
		}

		absolute_orientation(Xk, Yk, R, T);
	}

	void TrackingModule::absolute_orientation(cv::Mat1f &X, cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T) {

		cv::Matx31f meanX(0, 0, 0), meanY(0, 0, 0);

		int point_count = X.rows;

		// Calculate mean
		for (int i = 0; i < point_count; i++) {
			meanX(0) += X(i, 0);
			meanX(1) += X(i, 1);
			meanX(2) += X(i, 2);
			meanY(0) += Y(i, 0);
			meanY(1) += Y(i, 1);
			meanY(2) += Y(i, 2);
		}
		meanX *= 1.0f / point_count;
		meanY *= 1.0f / point_count;

		// Subtract mean
		for (int i = 0; i < point_count; i++) {
			X(i, 0) -= meanX(0);
			X(i, 1) -= meanX(1);
			X(i, 2) -= meanX(2);
			Y(i, 0) -= meanY(0);
			Y(i, 1) -= meanY(1);
			Y(i, 2) -= meanY(2);
		}

		// Rotation
		cv::Mat1f A;
		A = Y.t() * X;

		cv::SVD svd(A);

		cv::Mat1f Rmat;
		Rmat = svd.vt.t() * svd.u.t();
		Rmat.copyTo(R);

		// Translation
		T = meanX - R*meanY;
	}

	void TrackingModule::triangulate_matches(vector<DMatch>& matches, const vector<KeyPoint>&keypoints1, const vector<KeyPoint>& keypoints2,
		Mat& cam1P, Mat& cam2P, vector<Point3f>& pnts3D)
	{
		// Convert keypoints into Point2f
		vector<cv::Point2f> points1, points2;

		int i = 0;
		for (auto it = matches.begin(); it != matches.end(); ++it)
		{
			// Get the position of left keypoints
			float xl = keypoints1[it->queryIdx].pt.x;
			float yl = keypoints1[it->queryIdx].pt.y;

			points1.push_back(Point2f(xl, yl));

			// Get the position of right keypoints
			float xr = keypoints2[it->trainIdx].pt.x;
			float yr = keypoints2[it->trainIdx].pt.y;

			points2.push_back(Point2f(xr, yr));

			i++;
		}

		Mat pnts3DMat;
		cv::triangulatePoints(cam1P, cam2P, points1, points2, pnts3DMat);

		assert(pnts3DMat.cols == keypoints1.size());

		for (int x = 0; x < pnts3DMat.cols; x++) {
			float W = pnts3DMat.at<float>(3, x);
			float Z = pnts3DMat.at<float>(2, x) / W; /// 1000;

			if (fabs(Z - MAX_Z) < FLT_EPSILON || fabs(Z) > MAX_Z || Z < 0) {
				pnts3D.push_back(Point3f(0, 0, 0)); // Push empty point TODO: replace with lookup table?
				continue;
			}

			float X = pnts3DMat.at<float>(0, x) / W; /// 1000;
			float Y = pnts3DMat.at<float>(1, x) / W; /// 1000;

			pnts3D.push_back(Point3f(X, Y, Z));
		}

		assert(pnts3D.size() == keypoints1.size());
	}

};