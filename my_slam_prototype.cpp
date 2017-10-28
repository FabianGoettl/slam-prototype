#include <boost/thread/thread.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/xfeatures2d.hpp"

/*#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>*/

#include <iostream>
#include <stdio.h>
#include <cmath>

#include "my_slam_tracking.h"


using namespace cv;
using namespace std;
//using namespace pcl;

enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };

Size img_size = Size(640, 480);

// Stereo parameters
Mat Q;
Mat P1, P2;
Rect roi1, roi2;
Mat map11, map12, map21, map22;
float scale;

// Stereo matcher parameters
int alg = STEREO_SGBM;
int uniquenessRatio = 1;
int speckleWindowSize = 10;
int speckleRange = 80;

int SADWindowSize, numberOfDisparities, blockSize;

string calibration_filename = "";
string disparity_filename = "";
string point_cloud_filename = "";

/*

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVis(string windowName)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(windowName));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->setCameraPosition(0, 0, -1, 0, -1, 0);
	return (viewer);
}*/


static void print_help()
{
	printf("\nStereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
		"[--no-display] [-o=<disparity_image>] [-p=<point_cloud_file>]\n");
}





void filter_stereo_features(const vector<cv::DMatch>& matches, vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& goodMatches, double maxYDistance)
{
	if (matches.size() == 0) return;

	goodMatches.clear();

	for (vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		// Get the position of left keypoints
		float xl = keypoints1[it->queryIdx].pt.x;
		float yl = keypoints1[it->queryIdx].pt.y;

		// Get the position of right keypoints
		float xr = keypoints2[it->trainIdx].pt.x;
		float yr = keypoints2[it->trainIdx].pt.y;

		if (abs(yl - yr) <= maxYDistance) {
			goodMatches.push_back(*it);
		}
	}
}

void stereo_match(const Mat &img1, const Mat &img2, Mat &disparity, Mat &xyz) {
	numberOfDisparities = ((img_size.width / 8) + 15) & -16;
	//numberOfDisparities = 112;

	// Stereo matchers
	Ptr<StereoBM> bm = StereoBM::create(16, 9);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(61);
	bm->setBlockSize(blockSize > 0 ? blockSize : 9);
	bm->setMinDisparity(-39);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(507);
	bm->setUniquenessRatio(0);
	bm->setSpeckleWindowSize(0);
	bm->setSpeckleRange(8);
	bm->setDisp12MaxDiff(1);

	sgbm->setPreFilterCap(10);
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);

	int cn = img1.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(uniquenessRatio);
	sgbm->setSpeckleWindowSize(speckleWindowSize);
	sgbm->setSpeckleRange(speckleRange);
	sgbm->setDisp12MaxDiff(1);

	if (alg == STEREO_HH)
		sgbm->setMode(StereoSGBM::MODE_HH);
	else if (alg == STEREO_SGBM)
		sgbm->setMode(StereoSGBM::MODE_SGBM);
	else if (alg == STEREO_3WAY)
		sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

	Mat disp;
	//Mat img1p, img2p, dispp;
	//copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	//int64 t = getTickCount();

	if (alg == STEREO_BM)
		bm->compute(img1, img2, disp);
	else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
		sgbm->compute(img1, img2, disp);

	//t = getTickCount() - t;
	//printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

	//disp = dispp.colRange(numberOfDisparities, img1p.cols);
	if (alg != STEREO_VAR)
		disp.convertTo(disparity, CV_8U, 255 / (numberOfDisparities*16.));
	else
		disp.convertTo(disparity, CV_8U);

	cv::reprojectImageTo3D(disp, xyz, Q, true);
}


int main(int argc, char** argv)
{

	cv::CommandLineParser parser(argc, argv,
		"{@arg1||}{@arg2||}{help h||}{algorithm||}{max-disparity|0|}{blocksize|0|}{scale|1|}{i||}{e||}{o||}{p||}");
	
	if (parser.has("help"))
	{
		print_help();
		return 0;
	}

	if (parser.has("algorithm"))
	{
		string _alg = parser.get<string>("algorithm");
		alg = _alg == "bm" ? STEREO_BM :
			_alg == "sgbm" ? STEREO_SGBM :
			_alg == "hh" ? STEREO_HH :
			_alg == "var" ? STEREO_VAR :
			_alg == "sgbm3way" ? STEREO_3WAY : -1;
	}

	numberOfDisparities = parser.get<int>("max-disparity");
	SADWindowSize = parser.get<int>("blocksize");
	scale = parser.get<float>("scale");

	if (parser.has("i"))
		calibration_filename = parser.get<string>("i");
	if (parser.has("o"))
		disparity_filename = parser.get<string>("o");
	if (parser.has("p"))
		point_cloud_filename = parser.get<string>("p");

	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}
	if (alg < 0)
	{
		printf("Command-line parameter error: Unknown stereo algorithm\n\n");
		print_help();
		return -1;
	}
	if (numberOfDisparities % 16 != 0)
	{
		printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
		print_help();
		return -1;
	}
	if (scale < 0)
	{
		printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
		return -1;
	}
	if (SADWindowSize != 0 && SADWindowSize % 2 != 1)
	{
		printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
		return -1;
	}
	if (calibration_filename.empty())
	{
		printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to triangulate 3d points\n");
		return -1;
	}

	if (!calibration_filename.empty())
	{
		// reading intrinsic parameters
		FileStorage fs(calibration_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", calibration_filename.c_str());
			return -1;
		}

		fs["P1"] >> P1;
		fs["P2"] >> P2;
		fs["Q"] >> Q;

		Mat M1, D1, M2, D2;
		fs["CM1"] >> M1;
		fs["D1"] >> D1;
		fs["CM2"] >> M2;
		fs["D2"] >> D2;

		M1 *= scale;
		M2 *= scale;

		Mat R, T, R1, R2;
		fs["R"] >> R;
		fs["T"] >> T;
		fs["R1"] >> R1;
		fs["R2"] >> R2;

		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
	}


	VideoCapture capA, capB;

	if (!capA.open(0) || !capB.open(1))
		return 0;

	//namedWindow("disparity", 1);
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> scene_viewer;
	//scene_viewer = createVis("Scene");

	bool do_tracking;
	long frame_count = 0;
	bool initialized = false;

	Mat prev_img1;

	slam::TrackingSharedData shared;

	while (1)
	{
		Mat frameAraw, frameBraw, frameA, frameB;

		capA >> frameAraw;
		capB >> frameBraw;

		if (frameAraw.empty() || frameBraw.empty()) continue;

		frame_count++;

		do_tracking = frame_count > 10;

		if (!do_tracking) continue;

		// Remapping stereo image pair to share same epipolar lines
		Mat img1, img2;

		if (!calibration_filename.empty())
		{
			remap(frameAraw, img1, map11, map12, INTER_LINEAR);
			remap(frameBraw, img2, map21, map22, INTER_LINEAR);
		}
		else {
			img1 = frameAraw;
			img2 = frameBraw;
		}


		////////////////////////////////////

		// Extracting features
		/*Ptr<cv::ORB> detectorORB = ORB::create();
		vector<KeyPoint> keypoints1, keypoints2;
		Mat descriptors1, descriptors2;

		cout << "detectStart\n";
		detectorORB->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
		detectorORB->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
		cout << "detectEnd\n";

		// Match keypoints
		vector<DMatch> matches, goodMatches;

		if (keypoints1.size() == 0 || keypoints2.size() == 0) continue;

		cout << "matchStart\n";
		//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		//matcher->match(descriptors1, descriptors2, matches, noArray());
		//matchFeatures(descriptors1, descriptors2, matches);

		cout << "matchEnd\n";

		cout << "Cols: " << descriptors1.cols << endl;

		if (matches.size() == 0) continue;

		// Filter features along y-axis
		double maxYDistance = 10;
		filterFeatures(matches, keypoints1, keypoints2, goodMatches, maxYDistance);

		if (goodMatches.size() == 0) continue;

		// Draw stereo matches
		Mat res;
		cout << "drawStart\n";
		drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, res, Scalar(255, 0, 0), Scalar(255, 0, 0));
		imshow("stereo matches", res);
		cout << "drawEnd\n";*/
		

		/////////////// Time variant test /////////////////////

		/*if (prev_img1.empty()) {
			prev_img1 = img1.clone();
			continue;
		}

		Ptr<cv::ORB> detectorORB = ORB::create();
		vector<KeyPoint> keypoints1, keypoints2;
		Mat descriptors1, descriptors2;

		cout << "detectStart\n";
		detectorORB->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
		detectorORB->detectAndCompute(prev_img1, noArray(), keypoints2, descriptors2);
		cout << "detectEnd\n";

		// Match keypoints
		vector<DMatch> matches, goodMatches;

		if (keypoints1.size() == 0 || keypoints2.size() == 0) continue;

		cout << "matchStart\n";
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		matcher->match(descriptors1, descriptors2, matches, noArray());

		cout << "matchEnd\n";

		cout << "Cols: " << descriptors1.cols << endl;

		if (matches.size() == 0) continue;

		// Filter features along y-axis
		double maxYDistance = 10;
		filterFeatures(matches, keypoints1, keypoints2, goodMatches, maxYDistance);
		//goodMatches = matches;

		if (goodMatches.size() == 0) continue;

		// Draw stereo matches
		Mat res;
		cout << "drawStart\n";
		drawMatches(img1, keypoints1, prev_img1, keypoints2, goodMatches, res, Scalar(255, 0, 0), Scalar(255, 0, 0));
		imshow("stereo matches", res);
		cout << "drawEnd\n";

		prev_img1 = img1.clone();*/

		////////////////////////////////////



		// Extracting features
		Ptr<cv::ORB> detectorORB = ORB::create();
		vector<KeyPoint> keypoints1, keypoints2;
		Mat descriptors1, descriptors2;

		detectorORB->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
		detectorORB->detectAndCompute(img2, noArray(), keypoints2, descriptors2);


		// Match features (stereo)
		if (keypoints1.size() == 0 || keypoints2.size() == 0) continue;
		
		vector<DMatch> matches, goodMatches;

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		matcher->match(descriptors1, descriptors2, matches, noArray());

		if (matches.size() == 0) continue;

		// Filter features to share same epipolar line
		double maxYDistance = 10;
		filter_stereo_features(matches, keypoints1, keypoints2, goodMatches, maxYDistance);

		if (goodMatches.size() == 0) continue;

		// Draw stereo matches
		Mat res;
		drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, res, Scalar(255, 0, 0), Scalar(255, 0, 0));
		imshow("Stereo Matches", res);



		// Match features with tracks (temporal)
		std::vector<int> match_idx;
		slam::TrackingModule::match_features(shared.tracks, descriptors1, match_idx);

		cout << "Tracks:" << shared.tracks.size() << endl;


		// Draw active and tracked features
		Mat keypoints_img;

		std::vector<cv::KeyPoint> tracked_keypoints1;
		slam::TrackingModule::get_tracked_keypoints(keypoints1, tracked_keypoints1, match_idx);

		drawKeypoints(img1, keypoints1, keypoints_img, Scalar(0, 255, 0));
		drawKeypoints(keypoints_img, tracked_keypoints1, keypoints_img, Scalar(0, 0, 255));
		cv::imshow("Active and Tracked Keypoints", keypoints_img);

		// Triangulate stereo features
		std::vector<cv::Point3f> pnts3D;
		slam::TrackingModule::triangulate_matches(matches, keypoints1, keypoints2, P1, P2, pnts3D);

		// Update tracks
		slam::TrackingModule::update_tracks(shared.tracks, keypoints1, descriptors1, match_idx, pnts3D);

		// Stereo matching
		Mat disparity, xyz;
		stereo_match(img1, img2, disparity, xyz);

		// Draw disparity map
		Mat colorDisparity;
		applyColorMap(disparity, colorDisparity, cv::ColormapTypes::COLORMAP_JET);
		imshow("disparity", colorDisparity);


		//Create new view
		bool add_view = false;
		slam::TrackedView new_view;
		cv::Mat3f pointmap;
		if (shared.views.empty()) {
			new_view.R = shared.base_R;
			new_view.T = shared.base_T;
			add_view = true;
		}
		else {
			float movement = slam::TrackingModule::get_median_feature_movement(shared.tracks);

			cout << "Feature movement:" << movement << endl;

			if (movement > 100) {
				std::cout << "Movement is " << movement << "! Computing transformation...";

				cv::Matx33f stepR;
				cv::Matx31f stepT;
				slam::TrackingModule::transformation_from_tracks(shared.tracks,stepR, stepT);
				new_view.R = shared.base_R * stepR;
				new_view.T = shared.base_T + shared.base_R*stepT;
				add_view = true;
			}
		}

		if (add_view) {
			shared.views.push_back(new_view);
			shared.base_R = new_view.R;
			shared.base_T = new_view.T;

			shared.tracks.clear();

			for (unsigned int i = 0; i< keypoints1.size(); i++)
			{
				slam::FeatureTrack track;
				track.base_position = keypoints1[i].pt;
				track.base_position_3d = pnts3D[i];

				track.active_position = keypoints1[i].pt;
				track.active_position_3d = pnts3D[i];

				descriptors1.row(i).copyTo(track.descriptor);

				track.missed_frames = 0;

				shared.tracks.push_back(track);
			}
		}

		char key = waitKey(10);
		printf("\n");
	}

	return 0;
}
