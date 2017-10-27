#include <boost/thread/thread.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>

#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace cv;
using namespace std;

const string feature_cloud_name = "RGB point cloud";

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
}

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

int main(int argc, char** argv)
{
	/*pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_registered(new pcl::PointCloud<pcl::PointXYZ>);

	if (pcl::io::loadPCDFile<pcl::PointXYZ>("cloud0.pcd", *source) == -1) //* load the file
	{
		PCL_ERROR("Couldn't read file \n");
	}

	if (pcl::io::loadPCDFile<pcl::PointXYZ>("cloud1.pcd", *target) == -1) //* load the file
	{
		PCL_ERROR("Couldn't read file \n");
	}

	source->width = (int)source->points.size();
	source->height = 1;
	target->width = (int)target->points.size();
	target->height = 1;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->setCameraPosition(0, 0, -1, 0, -1, 0);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
		red_source(source, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(source, red_source, "source");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
		green_target(target, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target, green_target, "target");


	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	// Set the input source and target
	icp.setInputCloud(source);
	icp.setInputTarget(target);
	// Set the max correspondence distance (e.g., correspondences with higher distances will be ignored)
	icp.setMaxCorrespondenceDistance(0.5);
	// Set the maximum number of iterations (criterion 1)
	icp.setMaximumIterations(100000);
	icp.setRANSACIterations(10000);
	// Set the transformation epsilon (criterion 2)
	icp.setTransformationEpsilon(1e-5);
	// Set the euclidean distance difference epsilon (criterion 3)
	icp.setEuclideanFitnessEpsilon(0.01);
	// Perform the alignment
	icp.align(*source_registered);

	cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore() << endl;

	// Obtain the transformation that aligned cloud_source to cloud_source_registered
	Eigen::Matrix4f transformation = icp.getFinalTransformation();



	printf("ICP Transformation\n");
	cout << transformation << endl;

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
		white_source_registered(source_registered, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(source_registered, white_source_registered, "registered");

	while (1) {
		viewer->spinOnce(100);
	}*/

	boost::shared_ptr<pcl::visualization::PCLVisualizer> feature_viewer, registration_viewer;
	feature_viewer = createVis("Feature point cloud");
	registration_viewer = createVis("Registration point clouds");


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

	int savedClouds = 0;

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

	Eigen::Matrix4f cur_camera_pose = Eigen::Matrix4f::Identity();

	pcl::PointCloud<pcl::PointXYZ>::Ptr prev_frame_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr total_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

	//vector<DMatch> total_matches;
	//vector<Point3f> total_world_pts;

	Mat prev_img1, prev_img2;
	vector<cv::KeyPoint> prev_keypoints1, prev_keypoints2;
	Mat prev_descriptors1, prev_descriptors2;
	vector<Point3f> prev_world_pts;

	map<int, int> prev_keypointIdxToWorldIdxLookup;

	bool initialized = false;
	while (1)
	{
		Mat frameAraw, frameBraw, frameA, frameB;

		capA >> frameAraw;
		capB >> frameBraw;

		Mat img1, img2;

		if (!intrinsic_filename.empty())
		{
			remap(frameAraw, img1, map11, map12, INTER_LINEAR);
			remap(frameBraw, img2, map21, map22, INTER_LINEAR);
		}
		else {
			img1 = frameAraw;
			img2 = frameBraw;
		}

		// Extracting features
		Ptr<ORB> detectorORB = ORB::create();
		vector<KeyPoint> keypoints1, keypoints2;
		Mat descriptors1, descriptors2;

		detectorORB->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
		detectorORB->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

		// Match keypoints
		vector<DMatch> matches, goodMatches;
		vector<KeyPoint> matched1, matched2;

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		matcher->match(descriptors1, descriptors2, matches, noArray());

		double maxYDistance = 5;
		filterFeatures(matches, keypoints1, keypoints2, goodMatches, maxYDistance);

		if (goodMatches.size() == 0) continue;

		// Draw stereo matches
		Mat res;
		drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, res, Scalar(255, 0, 0), Scalar(255, 0, 0));
		imshow("keypoint matches", res);

		vector<Point3f> world_pts;
		map<int, int> keypointIdxToWorldIdxLookup;

		triangulateMatches(goodMatches, keypoints1, keypoints2, P1, P2, world_pts, keypointIdxToWorldIdxLookup);

		//assert(keypointIdxToWorldIdxLookup.size() == world_pts.size());

		/*pcl::PointCloud<pcl::PointXYZ>::Ptr cur_frame_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

		if (prev_keypoints1.empty() && prev_keypoints2.empty()) {
			
		}
		else {
			vector<DMatch> matches1, matches2;
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
			matcher->match(descriptors1, prev_descriptors1, matches1, noArray());
			//matcher->match(descriptors2, prev_descriptors2, matches2, noArray());

			// Draw previous matches
			Mat res;
			if (prev_keypoints1.size() == keypoints1.size()) {
				drawMatches(prev_img1, prev_keypoints1, img1, keypoints1, matches1, res, Scalar(255, 0, 0), Scalar(255, 0, 0));
				imshow("prev keypoint matches", res);
			}

			prev_frame_cloud_ptr->clear();

			// Turn matches with current and previous features into a point cloud
			for (auto it = matches1.begin(); it != matches1.end(); ++it)
			{
				if (keypointIdxToWorldIdxLookup.count(it->queryIdx) == 0) continue;

				// Get 3D point of current feature
				Point3f currentWorldPnt = world_pts[keypointIdxToWorldIdxLookup[it->queryIdx]];
				
				pcl::PointXYZ cloudPoint;
				cloudPoint.x = currentWorldPnt.x;
				cloudPoint.y = currentWorldPnt.y;
				cloudPoint.z = currentWorldPnt.z;

				cur_frame_cloud_ptr->points.push_back(cloudPoint);

				if (prev_keypointIdxToWorldIdxLookup.count(it->queryIdx) == 0) continue;

				// Get 3D point of previous feature
				Point3f prevWorldPnt = prev_world_pts[prev_keypointIdxToWorldIdxLookup[it->trainIdx]];

				cloudPoint.x = prevWorldPnt.x;
				cloudPoint.y = prevWorldPnt.y;
				cloudPoint.z = prevWorldPnt.z;

				prev_frame_cloud_ptr->points.push_back(cloudPoint);
			}
		}

		prev_keypoints1 = keypoints1;
		prev_keypoints2 = keypoints2;
		prev_descriptors1 = descriptors1;
		prev_descriptors2 = descriptors2;
		prev_img1 = img1.clone();
		prev_img2 = img2.clone();
		prev_world_pts = world_pts;
		prev_keypointIdxToWorldIdxLookup = keypointIdxToWorldIdxLookup;

		// Append to vectors
		//total_matches.insert(total_matches.end(), goodMatches.begin(), goodMatches.end());
		//total_world_pts.insert(total_world_pts.end(), worldPts.begin(), worldPts.end());
		*/
		// --- Create new point cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr cur_frame_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

		for (Point3f &p : world_pts) {
			pcl::PointXYZ cloudPoint;
			cloudPoint.x = p.x;
			cloudPoint.y = p.y;
			cloudPoint.z = p.z;

			cur_frame_cloud_ptr->points.push_back(cloudPoint);
		}


		// --- Draw keypoints
		/*for (KeyPoint &k : keypoints1) {
			int x = round(k.pt.x);
			int y = round(k.pt.y);

			circle(frameAraw, Point(x, y), 5, Scalar(0, 255, 0, 128));  //Scalar(B, G, R, 128)

			//cout << "Keypoint Depth: " << depth << endl;
		}
		for (KeyPoint &k : keypoints2) {
			int x = round(k.pt.x);
			int y = round(k.pt.y);

			circle(img2, Point(x, y), 5, Scalar(0, 255, 0, 128));
		}*/

		bool transformFound = false;
		if (prev_frame_cloud_ptr->points.size() > 0 && cur_frame_cloud_ptr->points.size() > 0) {

			pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
			// Set the input source and target
			icp.setInputCloud(cur_frame_cloud_ptr);
			icp.setInputTarget(prev_frame_cloud_ptr);
			// Set the max correspondence distance (e.g., correspondences with higher distances will be ignored)
			icp.setMaxCorrespondenceDistance(0.5);
			// Set the maximum number of iterations (criterion 1)
			icp.setMaximumIterations(100000);
			icp.setRANSACIterations(10000);
			// Set the transformation epsilon (criterion 2)
			icp.setTransformationEpsilon(1e-5);
			// Set the euclidean distance difference epsilon (criterion 3)
			icp.setEuclideanFitnessEpsilon(0.01);

			pcl::PointCloud<pcl::PointXYZ>::Ptr last_frame_registered(new pcl::PointCloud<pcl::PointXYZ>);

			// Perform the alignment
			icp.align(*last_frame_registered);



			double maxFitnessScore = 0.1;
			if (icp.hasConverged() && (icp.getFitnessScore() <= maxFitnessScore)) {

				transformFound = true;

				cout << "GOOD: ICP has converged:" << icp.hasConverged() << " score: " <<
					icp.getFitnessScore() << endl;

				// Obtain the transformation that aligned source to source_registered
				Eigen::Matrix4f transformation = icp.getFinalTransformation();
				printf("ICP Transformation\n");
				cout << transformation << endl;

				cur_camera_pose = transformation * cur_camera_pose;

				pcl::PointCloud<pcl::PointXYZ> cur_frame_cloud_transformed;
				pcl::transformPointCloud(*cur_frame_cloud_ptr, cur_frame_cloud_transformed, cur_camera_pose);

				// Add the point cloud data to the total cloud
				*total_cloud_ptr += cur_frame_cloud_transformed;

				// -- Show total features in viewer
				if (feature_viewer->contains(feature_cloud_name)) {
					feature_viewer->updatePointCloud(total_cloud_ptr, feature_cloud_name);
				}
				else {
					// Add point cloud
					total_cloud_ptr->width = (int)total_cloud_ptr->points.size();
					total_cloud_ptr->height = 1;

					pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
						green(total_cloud_ptr, 0, 255, 0);
					feature_viewer->addPointCloud<pcl::PointXYZ>(total_cloud_ptr, green, feature_cloud_name);
				}

				registration_viewer->removeAllPointClouds();

				cur_frame_cloud_ptr->width = (int)cur_frame_cloud_ptr->points.size();
				cur_frame_cloud_ptr->height = 1;
				prev_frame_cloud_ptr->width = (int)prev_frame_cloud_ptr->points.size();
				prev_frame_cloud_ptr->height = 1;
				last_frame_registered->width = (int)last_frame_registered->points.size();
				last_frame_registered->height = 1;

				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
					red_source(cur_frame_cloud_ptr, 255, 0, 0);
				registration_viewer->addPointCloud<pcl::PointXYZ>(cur_frame_cloud_ptr, red_source, "source");

				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
					green_target(prev_frame_cloud_ptr, 0, 255, 0);
				registration_viewer->addPointCloud<pcl::PointXYZ>(prev_frame_cloud_ptr, green_target, "target");

				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
					white_source_registered(last_frame_registered, 255, 255, 255);
				registration_viewer->addPointCloud<pcl::PointXYZ>(last_frame_registered, white_source_registered, "registered");


				registration_viewer->spinOnce(10);
				feature_viewer->spinOnce(100);
			}
			else {
				cout << "SKIPPED: Bad ICP:" << icp.hasConverged() << " score: " <<
					icp.getFitnessScore() << endl;
			}

			// Cleanup
			//last_frame_registered->clear();
		}

		if (!initialized || transformFound) {
			// Set current frame as last frame
			prev_frame_cloud_ptr->clear();
			prev_frame_cloud_ptr = cur_frame_cloud_ptr;
			initialized = true;
		}

		char key = waitKey(10);
		if (key == 's') {
			printf("Saving pointcloud...\n");
			pcl::io::savePCDFileASCII("cloud" + to_string(savedClouds++) + ".pcd", *cur_frame_cloud_ptr);
		}

		printf("\n");
	}

	return 0;
}
