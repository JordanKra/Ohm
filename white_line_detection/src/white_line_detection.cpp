#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cv_bridge/rgb_colors.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

/// Node that consumes images published on the 'image_raw' topic and republishes any
/// white pixels in the images to pointcloud/pointcloud2. This is used to detect the white lines
/// as an obstacle for Ohm to avoid.
class WhiteLineDetectionNode
{
private:
	/// Enable the openCv visualization if set by the node param.
	bool enableImshow{};

	/// Set on our first connection to the camera.
	bool connected{};

	/// Publish to pointcloud topics
	ros::Publisher pixelPub, pixelPubPCL2;
	/// Subscribe to the camera feed
	ros::Subscriber cameraInfoSub, imageSub;

	/// The lower and upper bound for what we define 'white' as.
	int lowColor, upperColor = 255, kernelSize, nthPixel;

	/// The per color bounds of what we define a 'white' pixel as.
	int lowB = lowColor, lowG = lowColor, lowR = lowColor;
	int highB = upperColor, highG = upperColor, highR = upperColor;

	/// Camera resolution as retrived from the camera_info topic.
	int HEIGHT{}, WIDTH{};

	/// Calibration constants. These are used for finding the offsets of the white pixels from the robot body.
	double A, B, C, D;

	/// The 3x3 perspective transform matrix. Should be treated as constant.
	cv::UMat Utransmtx;
	/// The region of intrest.
	const cv::Rect ROI = cv::Rect(112, 12, 1670 - 112, 920 - 12); // TODO: change to params. (x, y, width, height)

	/// Kernal used for white pixel filtering.
	cv::Mat erosionKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

	//empty callback functions but it is the only way to increment the sliders
	static void lowBlueTrackbar(int, void *) {}
	static void highBlueTrackbar(int, void *) {}
	static void lowGreenTrackbar(int, void *) {}
	static void highGreenTrackbar(int, void *) {}
	static void lowRedTrackbar(int, void *) {}
	static void highRedTrackbar(int, void *) {}

public:
	WhiteLineDetectionNode()
	{
		ros::NodeHandle nhPrivate("~");
		ros::NodeHandle nh;

		nhPrivate.param("calibration_constants/A", A, 0.0);
		nhPrivate.param("calibration_constants/B", B, 0.0);
		nhPrivate.param("calibration_constants/C", C, 0.0);
		nhPrivate.param("calibration_constants/D", D, 0.0);
		nhPrivate.param("sample_nth_pixel", nthPixel, 5);
		nhPrivate.param("kernel_size", kernelSize, 5);
		nhPrivate.param("lower_bound_white", lowColor, 160);
		nhPrivate.param("enable_imshow", enableImshow, false);

		// Activate main logic loop
		pixelPub = nh.advertise<sensor_msgs::PointCloud>("camera_cloud", 1);
		pixelPubPCL2 = nh.advertise<sensor_msgs::PointCloud2>("camera_cloud_pcl2", 1);

		imageSub = nh.subscribe<sensor_msgs::Image>("image_raw", 5, &WhiteLineDetectionNode::rosImageCallback, this);
		cameraInfoSub = nh.subscribe<sensor_msgs::CameraInfo>("camera_info", 1, &WhiteLineDetectionNode::rosCameraInfoCallback, this);
	}

	///Sets up the GPU to run our code using OpenCl.
	void setupOCL()
	{
		cv::setUseOptimized(true);
		cv::ocl::setUseOpenCL(true);
		if (cv::useOptimized())
		{
			std::cout << "OpenCL optimizations enabled" << std::endl;
		}
		else
		{
			std::cout << "OpenCL optimizations NOT enabled" << std::endl;
		}

		if (!cv::ocl::haveOpenCL())
		{
			std::cout << "no opencl devices detected" << std::endl;
		}

		cv::ocl::Context context;
		if (!context.create(cv::ocl::Device::TYPE_GPU))
		{
			std::cout << "failed to initialize device" << std::endl;
		}
		std::cout << context.ndevices() << " GPU device(s) detected." << std::endl;

		std::cout << "************************" << std::endl;
		for (size_t i = 0; i < context.ndevices(); i++)
		{
			cv::ocl::Device device = context.device(i);
			std::cout << "name: " << device.name() << std::endl;
			std::cout << "available: " << device.available() << std::endl;
			std::cout << "img support: " << device.imageSupport() << std::endl;
			std::cout << device.OpenCL_C_Version() << std::endl;
		}
		std::cout << "************************" << std::endl;

		cv::ocl::Device d = cv::ocl::Device::getDefault();
		std::cout << d.OpenCLVersion() << std::endl;
	}

	/// Sets up the constant perspective transform matrix as defined by node params.
	void setupWarp()
	{
		int tl_x, tl_y;
		int tr_x, tr_y;
		int br_x, br_y;
		int bl_x, bl_y;
		double ratio; // width / height of the actual panel on the ground

		ros::NodeHandle nhPrivate("~");

		nhPrivate.param("pixel_coordinates/top_left/x", tl_x, 0);
		nhPrivate.param("pixel_coordinates/top_left/y", tl_y, 0);
		nhPrivate.param("pixel_coordinates/top_right/x", tr_x, 0);
		nhPrivate.param("pixel_coordinates/top_right/y", tr_y, 0);
		nhPrivate.param("pixel_coordinates/bottom_right/x", br_x, 0);
		nhPrivate.param("pixel_coordinates/bottom_right/y", br_y, 0);
		nhPrivate.param("pixel_coordinates/bottom_left/x", bl_x, 0);
		nhPrivate.param("pixel_coordinates/bottom_left/y", bl_y, 0);
		nhPrivate.param("pixel_coordinates/ratio", ratio, 1.0);

		cv::Point Q1 = cv::Point2f(tl_x, tl_y); //top left pixel coordinate
		cv::Point Q2 = cv::Point2f(tr_x, tr_y); //top right
		cv::Point Q3 = cv::Point2f(br_x, br_y); //bottom right
		cv::Point Q4 = cv::Point2f(bl_x, bl_y); //bottom left

		double boardH = sqrt((Q3.x - Q2.x) * (Q3.x - Q2.x) + (Q3.y - Q2.y) * (Q3.y - Q2.y));
		double boardW = ratio * boardH;

		cv::Rect R(Q1.x, Q1.y, boardW, boardH);

		cv::Point R1 = cv::Point2f(R.x, R.y);
		cv::Point R2 = cv::Point2f(R.x + R.width, R.y);
		cv::Point R3 = cv::Point2f(cv::Point2f(R.x + R.width, R.y + R.height));
		cv::Point R4 = cv::Point2f(cv::Point2f(R.x, R.y + R.height));

		std::vector<cv::Point2d> squarePts{R1, R2, R3, R4};
		std::vector<cv::Point2d> quadPts{Q1, Q2, Q3, Q4};

		//Copy transform to constant feild
		auto transmtx = cv::getPerspectiveTransform(quadPts, squarePts);
		transmtx.copyTo(Utransmtx);
	}

	/// Converts a raw image to an openCv matrix. This function should decode the image properly automatically.
	///
	/// Returns the cv matrix form of the image.
	cv::UMat ptgrey2CVMat(const sensor_msgs::ImageConstPtr &imageMsg) const
	{
		auto cvImage = cv_bridge::toCvCopy(imageMsg, "CV_8UC3"); // This should decode correctly, but we may need to deal with bayer filters depending on the driver.
		return cvImage->image.getUMat(cv::ACCESS_RW);			 //TODO make sure this access is correct.
	}

	/// Applies the perspective warp to the image.
	///
	/// Returns the warped image matrix.
	cv::UMat shiftPerspective(cv::UMat &inputImage) const
	{
		//The transformed image
		auto transformed = cv::UMat(HEIGHT, WIDTH, CV_8UC1);
		//Apply the perspective warp previously calibrated.
		cv::warpPerspective(inputImage, transformed, Utransmtx, transformed.size());

		return transformed(ROI);
	}

	/// Filters non-white pixels out of the warped image.
	///
	/// Returns the eroded image matrix. The only pixels left should be white.
	cv::UMat imageFiltering(cv::UMat &warpedImage) const
	{
		auto binaryImage = cv::UMat(HEIGHT, WIDTH, CV_8UC1);
		auto erodedImage = cv::UMat(HEIGHT, WIDTH, CV_8UC1);

		cv::inRange(warpedImage, cv::Scalar(lowB, lowG, lowR), cv::Scalar(highB, highG, highR), binaryImage);
		cv::erode(binaryImage, erodedImage, erosionKernel);

		return erodedImage;
	}

	/// Converts the white pixel matrix into a pointcloud, then publishes the pointclouds.
	///
	/// This function works by offsetting each pixel by some calibrated constants to get the geographical lie of that pixel, which becomes a point
	/// in the pointcloud. This pointcloud is then broadcast, allowing the nav stack to see the white lines as obsticles.
	void getPixelPointCloud(cv::UMat &erodedImage) const
	{
		sensor_msgs::PointCloud msg;
		sensor_msgs::PointCloud2 msg2;
		std::vector<cv::Point> pixelCoordinates;

		cv::findNonZero(erodedImage, pixelCoordinates);
		//Iter through all the white pixels, adding their locations to pointclouds.
		for (size_t i = 0; i < pixelCoordinates.size(); i++)
		{
			if (i % nthPixel == 0)
			{
				geometry_msgs::Point32 pixelLoc;
				//XY distances of each white pixel relative to robot
				pixelLoc.x = (A * pixelCoordinates[i].x) + B;
				pixelLoc.y = (C * pixelCoordinates[i].y) + D;
				msg.points.push_back(pixelLoc);
			}
		}

		sensor_msgs::convertPointCloudToPointCloud2(msg, msg2);

		pixelPub.publish(msg);
		pixelPubPCL2.publish(msg2);
	}

	void createGUI()
	{
		cv::namedWindow("original", cv::WINDOW_FREERATIO);
		cv::namedWindow("erosion", cv::WINDOW_FREERATIO);
		cv::namedWindow("warp", cv::WINDOW_FREERATIO);
		cv::namedWindow("TRACKBARS", cv::WINDOW_FREERATIO);
		//*****************GUI related *********************************
		cv::createTrackbar("Low Blue", "TRACKBARS", &lowB, upperColor, lowBlueTrackbar);
		cv::createTrackbar("Low Green", "TRACKBARS", &lowG, upperColor, lowGreenTrackbar);
		cv::createTrackbar("Low Red", "TRACKBARS", &lowR, upperColor, lowRedTrackbar);
		cv::createTrackbar("High Blue", "TRACKBARS", &highB, upperColor, highBlueTrackbar);
		cv::createTrackbar("High Green", "TRACKBARS", &highG, upperColor, highGreenTrackbar);
		cv::createTrackbar("High Red", "TRACKBARS", &highR, upperColor, highRedTrackbar);
	}

	/// Updates the displayed guis with the last processed image.
	void display(cv::UMat &Uinput, cv::UMat &Utransformed, cv::UMat &Uerosion)
	{
		cv::imshow("original", Uinput);
		cv::imshow("warp", Utransformed);
		cv::imshow("erosion", Uerosion);
	}

	/// Callback passed to the image topic subscription. This produces a pointcloud for every
	/// image sent on the topic.
	void rosImageCallback(const sensor_msgs::ImageConstPtr &imageMsg)
	{
		if (!connected)
		{
			ROS_ERROR("Received image without receiving camera info first. This should not occur, and is a logic error.");
		}
		else
		{
			//Decode image
			auto cvImg = ptgrey2CVMat(imageMsg);
			//Perspective warp
			auto warpedImg = shiftPerspective(cvImg);
			//Filter non-white pixels out
			auto filteredImg = imageFiltering(warpedImg);
			//Convert pixels to pointcloud and publish
			getPixelPointCloud(filteredImg);

			//Display to gui if enabled.
			if (enableImshow)
				display(cvImg, warpedImg, filteredImg);
		}
	}

	/// Callback passed to the camera info sub.
	void rosCameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &infoMsg)
	{
		if (!connected) //attempt to prevent this callback from spamming.
		{
			HEIGHT = infoMsg->height;
			WIDTH = infoMsg->width;
			ROS_INFO("Connected to camera");
			connected = true;
		}
	}
};

int main(int argc, char **argv)
{
	ros::init(argc, argv, "white_line_detection");

	WhiteLineDetectionNode whiteLineDetector;

	//Setup enviornment
	whiteLineDetector.setupOCL();
	whiteLineDetector.createGUI();
	whiteLineDetector.setupWarp(); // 05/15/2019 starting from clockwise tl_x: (662, 315) (1263, 318) (1371, 522) (573, 520) 1.5015

	//This node is driven by the image topic, so just spin.
	ros::spin();

	ROS_INFO("Camera node shutting down");

	return 0;
}
