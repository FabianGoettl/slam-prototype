#include <boost/thread/thread.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <stdio.h>
#include <cmath>

#include "my_slam_tracking.h"

// Ratio to the second neighbor to consider a good match.
#define RATIO    0.6

using namespace cv;
using namespace std;


static void print_help()
{
	printf("\nStereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
		"[--no-display] [-o=<disparity_image>] [-p=<point_cloud_file>]\n");
}


int SADWindowSize, numberOfDisparities, blockSize;
int uniquenessRatio = 1;
int speckleWindowSize = 10;
int speckleRange = 80;

const int sad_slider_max = 30;
const int block_slider_max = 30;
const int uniquenessRatio_slider_max = 30;
const int speckleWindowSize_slider_max = 30;
const int speckleRange_slider_max = 30;

int blockSize_slider = 2;
int sad_slider = 3;
int uniquenessRatio_slider = uniquenessRatio;
int speckleWindowSize_slider = speckleWindowSize;
int speckleRange_slider = speckleRange;

list<slam::FeatureTrack> tracks; // feature tracks

void on_trackbar(int, void*)
{
	SADWindowSize = sad_slider;
	blockSize = 2 * blockSize_slider + 1;

	uniquenessRatio = uniquenessRatio_slider;
	speckleWindowSize = speckleWindowSize_slider;
	speckleRange = speckleRange_slider;
}


static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

void filterFeatures(const vector<cv::DMatch>& matches, vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& goodMatches, double maxYDistance)
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

void triangulateMatches(vector<DMatch>& matches, const vector<KeyPoint>&keypoints1, const vector<KeyPoint>& keypoints2, Mat& cam0P, Mat& cam1P, vector<Point3f>& pnts3D, map<int, int>& keypointIdxToWorldIdxLookup)
{
	map<int, int> worldIdxTokeypointIdxLookup;

	keypointIdxToWorldIdxLookup.clear();
	const double max_z = 3800; // 40*baseline

	// Convert keypoints into Point2f
	vector<cv::Point2f> pointsL, pointsR;

	int i = 0;
	for (auto it = matches.begin(); it != matches.end(); ++it)
	{
		// Get the position of left keypoints
		float xl = keypoints1[it->queryIdx].pt.x;
		float yl = keypoints1[it->queryIdx].pt.y;

		pointsL.push_back(Point2f(xl, yl));

		keypointIdxToWorldIdxLookup[it->queryIdx] = i;
		worldIdxTokeypointIdxLookup[i] = it->queryIdx;

		// Get the position of right keypoints
		float xr = keypoints2[it->trainIdx].pt.x;
		float yr = keypoints2[it->trainIdx].pt.y;

		pointsR.push_back(Point2f(xr, yr));

		i++;
	}

	Mat pnts3DMat;
	cv::triangulatePoints(cam0P, cam1P, pointsL, pointsR, pnts3DMat);

	for (int x = 0; x < pnts3DMat.cols; x++) {
		float W = pnts3DMat.at<float>(3, x);
		float Z = pnts3DMat.at<float>(2, x) / W / 1000;

		if (fabs(Z - max_z) < FLT_EPSILON || fabs(Z) > max_z || Z < 0) {
			keypointIdxToWorldIdxLookup.erase(worldIdxTokeypointIdxLookup[x]);
			pnts3D.push_back(Point3f(0, 0, 0));
			continue;
		}

		float X = pnts3DMat.at<float>(0, x) / W / 1000;
		float Y = pnts3DMat.at<float>(1, x) / W / 1000;

		pnts3D.push_back(Point3f(X, Y, Z));
	}
}




void matchFeatures(const cv::Mat &query, const cv::Mat &target,
	std::vector<cv::DMatch> &goodMatches) {
	std::vector<std::vector<cv::DMatch>> matches;
	FlannBasedMatcher matcher(new cv::flann::LshIndexParams(12, 20, 2));
	// Find 2 best matches for each descriptor to make later the second neighbor test.
	matcher.knnMatch(query, target, matches, 2);
	// Second neighbor ratio test.
	for (unsigned int i = 0; i < matches.size(); ++i) {
		if (matches[i].size() >= 2) {
			if (matches[i][0].distance < matches[i][1].distance * RATIO)
				goodMatches.push_back(matches[i][0]);
		}
	}
}


int main(int argc, char** argv)
{
	string img1_filename = "";
	string img2_filename = "";
	string intrinsic_filename = "";
	string extrinsic_filename = "";
	string disparity_filename = "";
	string point_cloud_filename = "";

	enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };
	int alg = STEREO_SGBM;

	bool no_display;
	float scale;

	Ptr<StereoBM> bm = StereoBM::create(16, 9);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
	cv::CommandLineParser parser(argc, argv,
		"{@arg1||}{@arg2||}{help h||}{algorithm||}{max-disparity|0|}{blocksize|0|}{no-display||}{scale|1|}{i||}{e||}{o||}{p||}");
	if (parser.has("help"))
	{
		print_help();
		return 0;
	}
	img1_filename = parser.get<string>(0);
	img2_filename = parser.get<string>(1);
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
	no_display = parser.has("no-display");
	if (parser.has("i"))
		intrinsic_filename = parser.get<string>("i");
	if (parser.has("e"))
		extrinsic_filename = parser.get<string>("e");
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
	if (img1_filename.empty() || img2_filename.empty())
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		return -1;
	}
	if ((intrinsic_filename.empty()))
	{
		printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
		return -1;
	}

	if (extrinsic_filename.empty() && !point_cloud_filename.empty())
	{
		printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
		return -1;
	}

	Size img_size = Size(640, 480);
	Rect roi1, roi2;
	Mat Q;
	Mat map11, map12, map21, map22;

	Mat P1, P2;

	if (!intrinsic_filename.empty())
	{
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename.c_str());
			return -1;
		}

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
		fs["P1"] >> P1;
		fs["P2"] >> P2;

		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
	}




	VideoCapture capA, capB;

	if (!capA.open(0) || !capB.open(1))
		return 0;

	bool do_tracking;
	int frame_count = 0;
	bool initialized = false;

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
		cout << "remapStart\n";
		if (!intrinsic_filename.empty())
		{
			remap(frameAraw, img1, map11, map12, INTER_LINEAR);
			remap(frameBraw, img2, map21, map22, INTER_LINEAR);
		}
		else {
			img1 = frameAraw;
			img2 = frameBraw;
		}
		cout << "remapEnd\n";

		// Extracting features
		Ptr<cv::ORB> detectorORB = ORB::create();
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
		matchFeatures(descriptors1, descriptors2, matches);
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
		imshow("keypoint matches", res);
		cout << "drawEnd\n";
		

		if (!initialized) {
			initialized = true;
		}

		char key = waitKey(10);
		printf("\n");
	}

	return 0;
}
