#include "DualEllipseFitting.h"
#include "RegionDetect.h"
#include "PointEllipseFitting.h"
#include "FeatureDetector.h"
#include "ZZYCalibration.h"
#include "SpaceCircleSolver.h"
#include "SpaceLineSolver.h"
#include "StereoCalibration.h"
#include "LineScanCalibration.h"
#include "PnPSolver.h"
#include "CommonFunctions.h"
#include "ZernikeMoment.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>

using std::normal_distribution;
using namespace MtPnPSolver;
using namespace MtZernikeMoment;
using namespace MtRegionDetect;
using namespace MtFeatureDetector;
using namespace MtDualEllipseFitting;
using namespace MtEllipseFitting;
using namespace MtZZYCalibration;
using namespace MtSpaceCircleSolver;
using namespace MtSpaceLineSolver;
using namespace MtStereoCalibration;

/*面阵相机标定程序*/
bool findCirclesCenter(const cv::Mat& img, const cv::Size& board_size, const cv::Size& square_size, std::vector<cv::Point2f>& corners)
{
    cv::Size image_size;

    image_size.width = img.cols;
    image_size.height = img.rows;
    
    cv::SimpleBlobDetector::Params params;
    params.minThreshold = 30;
    params.maxThreshold = 200;
    params.thresholdStep = 10;
    params.filterByInertia = true;
    params.filterByColor = true;
    params.blobColor = 0;
    params.filterByArea = true;
    params.minArea = 60;
    params.minDistBetweenBlobs = 15;
    
    cv::Ptr<cv::FeatureDetector> blobDetector = cv::SimpleBlobDetector::create(params);

    // extract circle center
    bool detectorFlag = findCirclesGrid(img, board_size, corners, cv::CALIB_CB_SYMMETRIC_GRID, blobDetector);

    if (detectorFlag == false)
    {
        cv::imshow("Failure", img);
        cv::waitKey(1000);
        return false;
    }

    return true;   
}

void testHomography(void)
{
    std::vector<std::string> files =
    {
        "E:\\Code\\EllipseFitSource\\data\\calibration\\left01.jpg",
        "E:\\Code\\EllipseFitSource\\data\\calibration\\left04.jpg"
    };

    std::vector<std::vector<cv::Point2f>> allCorners;
    cv::Size boardSize(6, 9);
    for (int i = 0; i < files.size(); i++)
    {
        cv::Mat img = cv::imread(files[i]);
        std::vector<cv::Point2f> corners;

        bool ok = cv::findChessboardCorners(img, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (ok) 
        {
            allCorners.push_back(corners);
            cv::drawChessboardCorners(img, boardSize, cv::Mat(corners), ok);
            cv::imshow("corners", img);
            cv::waitKey(0);
        }
    }

    std::vector<Eigen::Vector2d> srcPoints, dstPoints;

    for (cv::Point2f& pt : allCorners[0])
    {
        srcPoints.push_back(Eigen::Vector2d(pt.x, pt.y));
    }
    for (cv::Point2f& pt : allCorners[1])
    {
        dstPoints.push_back(Eigen::Vector2d(pt.x, pt.y));
    }
    
    std::cout << srcPoints.size() << std::endl;
    std::cout << dstPoints.size() << std::endl;
    Eigen::Matrix3d H;
    Eigen::Matrix3d H_CV;

    std::string cameraParaPath = "E:\\Code\\EllipseFitSource\\config\\config.yaml";
    ZZYCalibration calibration(5,files.size(),cameraParaPath);
    bool ok = calibration.findHomography(srcPoints, dstPoints, H, true);
    bool okcv = calibration.findHomographyByOpenCV(srcPoints, dstPoints, H_CV);

    if (ok)
    {
        std::cout << "H: " << H << std::endl;
        std::cout << "H_CV: " << H_CV << std::endl;
        cv::Mat srcImage = cv::imread(files[0]);
        cv::Mat dstImage = cv::imread(files[1]);
        cv::Mat result;
        cv::Mat Hmat;
        cv::eigen2cv(H, Hmat);
        cv::warpPerspective(srcImage, result, Hmat, dstImage.size());
        cv::imshow("result", result);
        cv::waitKey(0);
        cv::imwrite("result.jpg", result);
    }
}

void CalibrationWithChessBoard(const std::string& workingDir, cv::Size& imageSize, cv::Mat& K,cv::Mat& distCoeffs, std::vector<cv::Mat>& RvecsMat, std::vector<cv::Mat>& tvecsMat)
{
	// 运行函数必要的参数
	const int calibrationBoardRow = 8;
	const int calibrationBoardCol = 11;
	const float calibrationBoardXPitch = 20.;
	const float calibrationBoardYPitch = 20.;
	const int distortionParameterNum = 4;
	const cv::Size boardSize = cv::Size(calibrationBoardCol, calibrationBoardRow);
	const std::string camaraParaPath = "E:\\Code\\EllipseFitSource\\config\\config.yaml";

	// 提取图像特征点
	bool flag = true;
	std::vector<std::string> filePaths;
	cv::glob(workingDir, filePaths, false);
	const int imageCnt = filePaths.size();
	std::vector<std::vector<cv::Point2f>> imagePointsAll;
	for (int it = 0; it < imageCnt; ++it)
	{
		std::vector<cv::Point2f> imagePoints;
		cv::Mat imageInput = cv::imread(filePaths[it]);
		if (flag)
		{
			flag = false;
			imageSize.height = imageInput.rows;
			imageSize.width = imageInput.cols;
		}
		bool ok = cv::findChessboardCorners(imageInput, boardSize, imagePoints);
		if (!ok)
		{
			std::cout << "Error Extract In " << filePaths[it] << std::endl;
			std::cout << "Please Erase it And Continue..." << std::endl;
		}
		else
		{
			cv::Mat grayImage;
			if (imageInput.channels() == 3)
				cv::cvtColor(imageInput, grayImage, cv::COLOR_RGB2GRAY);
			else 
				grayImage = imageInput.clone();
			cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 20, 0.01);
			cv::cornerSubPix(grayImage, imagePoints, cv::Size(11, 11), cv::Size(-1, -1), criteria);
			imagePointsAll.push_back(imagePoints);

			/*cv::drawChessboardCorners(grayImage, boardSize, imagePoints, true);
			cv::imshow("chessBoard", grayImage);
			cv::waitKey(0);*/
		}
	}

	// 保存棋盘格三维信息
	std::vector<std::vector<cv::Point3f>> objectPointsAll;
	for (int it = 0; it < imageCnt; ++it)
	{
		std::vector<cv::Point3f> objectPoints;
		for (int i = 0; i < boardSize.height; ++i)
		{
			for (int j = 0; j < boardSize.width; ++j)
			{
				cv::Point3f realPoint;
				realPoint.x = i*calibrationBoardXPitch;
				realPoint.y = j*calibrationBoardYPitch;
				realPoint.z = 0.;
				objectPoints.push_back(realPoint);
			}
		}
		objectPointsAll.push_back(objectPoints);
	}

	// 开始相机标定流程
	std::vector<cv::Mat> rvecsMat;
	double err = cv::calibrateCamera(objectPointsAll, imagePointsAll, imageSize, K, distCoeffs, rvecsMat, tvecsMat, cv::CALIB_FIX_K3);
	

	// 开始评价标定结果
	std::vector<cv::Point2f> imagePointsPro;
	for (int it = 0; it < imageCnt; ++it)
	{
		cv::projectPoints(objectPointsAll[it], rvecsMat[it], tvecsMat[it], K, distCoeffs, imagePointsPro);
		double errPerImage = cv::norm(imagePointsAll[it], imagePointsPro, cv::NORM_L2);
		errPerImage /= objectPointsAll[it].size();
		std::cout << "Image " << it << " Reprojection Error: " << errPerImage << std::endl;
	}

	// 打印相机标定结果
	std::cout << std::endl << std::endl;
	std::cout << "fx: " << setprecision(8) << K.at<double>(0, 0) << std::endl;
	std::cout << "fy: " << setprecision(8) << K.at<double>(1, 1) << std::endl;
	std::cout << "cx: " << setprecision(8) << K.at<double>(0, 2) << std::endl;
	std::cout << "cy: " << setprecision(8) << K.at<double>(1, 2) << std::endl;
	std::cout << std::endl << std::endl;

	// 将旋转向量转化为旋转矩阵形式
	for (int it = 0; it < imageCnt; ++it)
	{
		cv::Mat RvecMat = cv::Mat(3, 3, CV_32FC1, cv::Scalar(0));
		cv::Rodrigues(rvecsMat[it], RvecMat);
		RvecsMat.push_back(RvecMat);
	}

	//system("pause");
}

void GetStereoExtrincMatrix(const std::vector<cv::Mat>& Rl, const std::vector<cv::Mat>& tl,
	const std::vector<cv::Mat>& Rr, const std::vector<cv::Mat>& tr,
	cv::Mat& R,cv::Mat& t)
{
	const int size = Rl.size();
	std::vector<cv::Mat> Rvecs;
	std::vector<cv::Mat> tvecs;
	for (int it = 0; it < size; ++it)
	{
		cv::Mat Rtmp = Rl[it] * Rr[it].t();
		cv::Mat ttmp = tr[it] - tl[it];
		
		Rvecs.push_back(Rtmp);
		tvecs.push_back(ttmp);
	}

	// 归一化平移向量
	float tVal = 0.;
	std::vector<cv::Mat> rvecs;
	cv::Mat tnorm = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0.));
	for (int it = 0; it < size; ++it)
	{
		cv::Mat r;
		double tmp = CommonFunctions::norm(tvecs[it]);
		tnorm += CommonFunctions::normalize(tvecs[it]);
		cv::Rodrigues(Rvecs[it], r);
		tVal += tmp; rvecs.push_back(r);
		std::cout << "t: " << tmp << std::endl;
	}
	tVal /= float(size);
	tnorm = CommonFunctions::normalize(tnorm);
	std::cout << std::endl << std::endl;
	std::cout << "baseline: " << setprecision(8) << tVal << std::endl;
	std::cout << std::endl << std::endl;

	// 归一化旋转向量
	float theta = 0.;
	cv::Mat rnorm = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0.));
	for (int it = 0; it < size; ++it)
	{
		double tmp = CommonFunctions::norm(rvecs[it]);
		rnorm += CommonFunctions::normalize(rvecs[it]);
		theta += tmp;
		std::cout << "theta: " << theta << std::endl;
	}
	rnorm = CommonFunctions::normalize(rnorm);
	theta /= float(size);

	cv::Rodrigues(rnorm*theta,R);
	t = tnorm*tVal;
}

void constructStereoImage(const cv::Mat& left, const cv::Mat& right, cv::Mat& all)
{
	const int width = left.cols;
	const int height = right.rows;
	all = cv::Mat(height, width * 2, left.type());

	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			all.at<uchar>(row, col) = left.at<uchar>(row, col);
		}
	}

	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			all.at<uchar>(row, col+width) = right.at<uchar>(row, col);
		}
	}
}

void RectifyingImage(void)
{
	const std::string workingDirLeft = "F:\\Users\\Admin\\Desktop\\left\\";
	const std::string workingDirRight = "F:\\Users\\Admin\\Desktop\\right\\";
	const std::string savingLeftImage = "F:\\Users\\Admin\\Desktop\\left.jpg";
	const std::string savingRightImage = "F:\\Users\\Admin\\Desktop\\right.jpg";
	cv::Size imageSize;
	cv::Mat KLeft, KRight;
	cv::Mat distCoeffsLeft, distCoeffsRight;
	std::vector<cv::Mat> RvecMatLeft, tvecMatLeft;
	std::vector<cv::Mat> RvecMatRight, tvecMatRight;
	
	// 分别标定双目相机
	CalibrationWithChessBoard(workingDirLeft,imageSize, KLeft, distCoeffsLeft, RvecMatLeft, tvecMatLeft);
	CalibrationWithChessBoard(workingDirRight,imageSize, KRight, distCoeffsRight, RvecMatRight, tvecMatRight);

	cv::Mat mapLx, mapLy, mapRx, mapRy;
	cv::Mat R, T, Rl, Rr, Pl, Pr, Q;
	GetStereoExtrincMatrix(RvecMatLeft, tvecMatLeft, RvecMatRight, tvecMatRight,R,T);
	cv::stereoRectify(KLeft, distCoeffsLeft, KRight, distCoeffsRight, imageSize, R, T, Rl, Rr, Pl, Pr, Q);
	cv::initUndistortRectifyMap(KLeft, distCoeffsLeft, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	cv::initUndistortRectifyMap(KRight, distCoeffsRight, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	cv::Mat leftImageGray, rightImageGray;
	cv::Mat rectifyLeftImage, rectifyRightImage;
	std::string leftImageFile = "E:\\Code\\Dcamera\\C++\\ApplicationExampleCode\\CaptureBinocularImages\\UsbCameraDemo\\UsbCameraDemo\\left\\ImageL000000.jpg";
	std::string rightImageFile = "E:\\Code\\Dcamera\\C++\\ApplicationExampleCode\\CaptureBinocularImages\\UsbCameraDemo\\UsbCameraDemo\\right\\ImageR000000.jpg";
	leftImageGray = cv::imread(leftImageFile, cv::IMREAD_COLOR);
	rightImageGray = cv::imread(rightImageFile, cv::IMREAD_COLOR);
	cv::remap(leftImageGray, rectifyLeftImage, mapLx, mapLy, cv::INTER_LINEAR);
	cv::remap(rightImageGray, rectifyRightImage, mapRx, mapRy, cv::INTER_LINEAR);

	cv::imwrite(savingLeftImage, rectifyLeftImage);
	cv::imwrite(savingRightImage, rectifyRightImage);

	cv::cvtColor(leftImageGray, leftImageGray, cv::COLOR_RGB2GRAY);
	cv::cvtColor(rightImageGray, rightImageGray, cv::COLOR_RGB2GRAY);
	cv::cvtColor(rectifyLeftImage, rectifyLeftImage, cv::COLOR_RGB2GRAY);
	cv::cvtColor(rectifyRightImage, rectifyRightImage, cv::COLOR_RGB2GRAY);

	cv::Mat leftRightImage, rectifyLeftRightImage;
	constructStereoImage(leftImageGray, rightImageGray, leftRightImage);
	constructStereoImage(rectifyLeftImage, rectifyRightImage, rectifyLeftRightImage);

	// 画上直线
	for (int it = 0; it < rectifyLeftRightImage.rows; it+=50)
	{
		cv::line(leftRightImage, cv::Point(0, it), cv::Point(rectifyLeftRightImage.cols-1, it),cv::Scalar(0,255,0));
		cv::line(rectifyLeftRightImage, cv::Point(0, it), cv::Point(rectifyLeftRightImage.cols-1, it),cv::Scalar(0,255,0));
	}
	cv::imshow("origin", leftRightImage);
	cv::imshow("rectyfy", rectifyLeftRightImage);
	cv::waitKey(0);
}

void testCalibration(void)
{
	// 运行函数必要的输入参数
	const bool fitRotationAxis = true;
	const bool showUndisImage = false;
	const int calibrationBoardRow = 9;
	const int calibrationBoardCol = 11;
	const float calibrationBoardXPitch = 4.2;
	const float calibrationBoardYPitch = 4.2;
	const int distortionParameterNum = 4;
	const int calibrationBoardFeaturesSize = calibrationBoardRow*calibrationBoardCol;
	const std::string workingDir = "E:\\Code\\EllipseFitSource\\data\\03\\";
    const std::string cameraParaPath = "E:\\Code\\EllipseFitSource\\config\\config.yaml";
	
	std::vector<std::string> filePaths;
	cv::glob(workingDir, filePaths, false);
	ZZYCalibration calibration(distortionParameterNum, filePaths.size(), cameraParaPath);
    std::vector<std::vector<Eigen::Vector2d>> imagePoints;
    std::vector<std::vector<Eigen::Vector3d>> objectPoints;
    cv::Size boardSize(calibrationBoardRow, calibrationBoardCol);
    cv::Size2f squareSize(calibrationBoardXPitch, calibrationBoardYPitch);
    for (int i = 0; i < filePaths.size(); ++i) 
    {
		std::cout << "Start Extract Features In " << filePaths[i] << std::endl;
        cv::Mat img = cv::imread(filePaths[i]);
        std::vector<cv::Point2d> cornersd;
		std::vector<cv::Point2f> cornersf;
		const int cornersSize = cornersd.size();
		cornersf.resize(cornersSize);
		for (int it = 0; it < cornersSize; ++it)
		{
			cornersf[it].x = cornersd[it].x;
			cornersf[it].y = cornersd[it].y;
		}

        FeatureDetector detect(img, calibrationBoardRow, calibrationBoardCol);
        detect.compute(cornersd);
		if (cornersd.size() != calibrationBoardFeaturesSize)
		{
			std::cout << "Error Image: " << filePaths[i] << std::endl;
			std::cout << "Please Erase The Error Image And Continue..." << std::endl;
		}
		std::vector<Eigen::Vector2d> _corners; 
		for (auto& pt : cornersd)
			_corners.push_back(Eigen::Vector2d(pt.x, pt.y));
		imagePoints.push_back(_corners);
		std::cout << "End Extract Features In " << filePaths[i] << std::endl;
    }
    
	// 开始相机标定流程
    for (int i = 0; i < imagePoints.size(); ++i) 
    {
        std::vector<Eigen::Vector3d> corners;
        calibration.getObjectPoints1(boardSize, squareSize, corners);
        objectPoints.push_back(corners);
    }

    calibration.setObjectPoints(objectPoints);
    calibration.setImagePoints(imagePoints);

    Eigen::Matrix3d CameraMatrix;
    Eigen::Vector3d RadialDistortion;
    Eigen::Vector2d TangentialDistortion;
    std::vector<std::vector<Eigen::Vector3d>> vX3D;
    calibration.compute(CameraMatrix, RadialDistortion, TangentialDistortion);
    calibration.get3DPoint(vX3D);

    if (fitRotationAxis)
    {
        const std::string filename = "./result/3D.pcd";
		const std::string fileOne3DTarget = "./result/OneTarget.pcd";
        const std::string fileRotationAxis = "./result/RotationAxis.pcd";
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr RotationAxis(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr OneTarget(new pcl::PointCloud<pcl::PointXYZ>);

        const int imgNum = vX3D.size();
        const int feaNum = vX3D[0].size();
        std::vector<Eigen::Vector3d> vSpaceLineData;

		for (int i = 0; i < feaNum; ++i)
		{
			pcl::PointXYZ point;
			point.x = vX3D[0][i](0);
			point.y = vX3D[0][i](1);
			point.z = vX3D[0][i](2);
			OneTarget->points.push_back(point);
		}

		OneTarget->height = 1;
		OneTarget->width = OneTarget->size();
		pcl::io::savePCDFile(fileOne3DTarget, *OneTarget);

        for (int i = 0; i < feaNum; i++)
        {
            int r = rand() % 255;
            int g = rand() % 255;
            int b = rand() % 255;

            pcl::PointXYZ axisPoint;
            Eigen::Vector3d spaceLinePoint;
            std::vector<Eigen::Vector3d> data;

            for (int j = 0; j < imgNum; j++)
            {
                pcl::PointXYZRGB point;
                Eigen::Vector3d epoint;
                point.x = vX3D[j][i](0);
                point.y = vX3D[j][i](1);
                point.z = vX3D[j][i](2);

                epoint(0) = vX3D[j][i](0);
                epoint(1) = vX3D[j][i](1);
                epoint(2) = vX3D[j][i](2);

                point.r = r;
                point.b = b;
                point.g = g;

                data.push_back(epoint);
                pointcloud->points.push_back(point);
            }

            // 开始空间圆拟合流程
            double radius;
            Eigen::VectorXd center;
            SpaceCircleSolver solver(data, 0.02, false);
            solver.compute(center, radius);

            axisPoint.x = center(0);
            axisPoint.y = center(1);
            axisPoint.z = center(2);

            spaceLinePoint(0) = center(0);
            spaceLinePoint(1) = center(1);
            spaceLinePoint(2) = center(2);

            RotationAxis->points.push_back(axisPoint);
            vSpaceLineData.push_back(spaceLinePoint);

			std::cout << "center: " << center.transpose() << std::endl;
        }

        pointcloud->height = 1;
        pointcloud->width = pointcloud->size();
        pcl::io::savePCDFile(filename, *pointcloud);

        RotationAxis->height = 1;
        RotationAxis->width = RotationAxis->size();
        pcl::io::savePCDFile(fileRotationAxis, *RotationAxis);

        // 开始空间直线拟合流程
        Eigen::Vector3d normal;
        SpaceLineSolver solver(vSpaceLineData, 0.2, false);
        solver.compute(normal);
        solver.RMS();

        std::cout << "SpaceLineDirection: " << normal.transpose() << std::endl;
    }
	
	if (showUndisImage)
	{
		cv::Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
		cv::eigen2cv(CameraMatrix, cameraMatrix);

		cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
		distCoeffs.at<double>(0) = RadialDistortion(0);
		distCoeffs.at<double>(1) = RadialDistortion(1);
		distCoeffs.at<double>(2) = TangentialDistortion(0);
		distCoeffs.at<double>(3) = TangentialDistortion(1);
		distCoeffs.at<double>(4) = RadialDistortion(2);

		cv::Mat undistImag;
		for (int i = 0; i < filePaths.size(); ++i)
		{
			cv::Mat img = cv::imread(filePaths[i]);
			cv::Mat gray;
			cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
			cv::Mat new_matrix;
			cv::undistort(gray, undistImag, cameraMatrix, distCoeffs, new_matrix);
			cv::namedWindow("undistortion", cv::WINDOW_KEEPRATIO);
			cv::namedWindow("image", cv::WINDOW_KEEPRATIO);
			cv::imshow("image", gray);
			cv::imshow("undistortion", undistImag);
			cv::waitKey(0);
		}
		cv::destroyAllWindows();
	}
	system("pause");
}

void generate_ellipse()
{
    int WINDOW_WIDTH = 600; 
    int thickness = 2;
    int lineType = 8;
    cv::Mat atomImage = cv::Mat::zeros(WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3);
    
    for (int i = 0; i < 12; i++)
    {
        for (int row = 0; row < WINDOW_WIDTH; row++)
        {
            uchar* imagedata = atomImage.ptr<uchar>(row);
            for (int col = 0; col < WINDOW_WIDTH; col++)
            {
                imagedata[0] = 255;
                imagedata[1] = 255;
                imagedata[2] = 255;
                imagedata += 3;
            }
        }

        cv::ellipse(atomImage, cv::Point(WINDOW_WIDTH / 2, WINDOW_WIDTH / 2),
            cv::Size(WINDOW_WIDTH / 4, WINDOW_WIDTH / 8), (i+1)*30, 0, 360, cv::Scalar(255, 129, 0),
            thickness, lineType);
        cv::imwrite("E:\\Code\\EllipseFitSource\\data\\ellipse"+to_string(i)+".jpg", atomImage);
    }
}

// calculate gradient with Prewitt operator
cv::Mat Prewitt(const cv::Mat& image, const string& flag)
{
    cv::Mat gradient(image.rows, image.cols, image.type());
    if (flag == "x")
    {
        const int Row = image.rows;
        const int Col = image.cols;
        for (int row = 1; row < Row - 1; row++)
        {
            for (int col = 1; col < Col - 1; col++)
            {
                gradient.at<uchar>(row, col) = image.at<uchar>(row - 1, col + 1) + image.at<uchar>(row, col + 1) + image.at<uchar>(row + 1, col + 1)
                    - image.at<uchar>(row - 1, col - 1) - image.at<uchar>(row, col - 1) - image.at<uchar>(row + 1, col - 1);
            }
        }
    }
    else if (flag == "y")
    {
        const int Row = image.rows;
        const int Col = image.cols;
        for (int row = 1; row < Row - 1; row++)
        {
            for (int col = 1; col < Col - 1; col++)
            {
                gradient.at<uchar>(row, col) = image.at<uchar>(row - 1, col - 1) + image.at<uchar>(row - 1, col) + image.at<uchar>(row - 1, col + 1)
                    - image.at<uchar>(row + 1, col - 1) - image.at<uchar>(row + 1, col) - image.at<uchar>(row + 1, col + 1);
            }
        }
    }
    return gradient;
}

void test_ellipse_fit(void)
{
    cv::Mat source;
    cv::Mat source_gray;
    source = cv::imread("E:\\Code\\EllipseFitSource\\data\\test2.jpg");
    cv::cvtColor(source, source_gray, cv::COLOR_RGB2GRAY);

    cv::Mat dx = Prewitt(source_gray, "x");
    cv::Mat dy = Prewitt(source_gray, "y");

    cv::imshow("GX", dx);
    cv::imshow("GY", dy);
    cv::imshow("origon", source_gray);

    const int Row = dx.rows;
    const int Col = dy.cols;

    std::vector<float> X;
    std::vector<float> Y;

    for (int row = 1; row < Row-1; row++)
    {
        for (int col = 1; col < Col-1; col++)
        {
            const int grayX = dx.at<uchar>(row, col);
            const int grayY = dy.at<uchar>(row, col);
            if (grayX != 0 || grayY != 0)
            {
                float coordinateX = row;
                float coordinateY = col;
                
                X.push_back(coordinateX);
                Y.push_back(coordinateY);
            }
        }
    }
    
    cv::Mat1d para;
    MtEllipseFitting::EllipseSolver fit(X, Y, false);
    fit.compute();
    fit.getEllipsePara(para);

    cv::ellipse(source, cv::Point(para(1, 0), para(0, 0)),
        cv::Size(para(2, 0), para(3, 0)),
        para(4, 0), 0, 360, cv::Scalar(255,129,0), 2, 8);

    imshow("result", source);
    cv::waitKey(0);
    
}

void testDualEllipse(void)
{
    cv::Mat source;
    cv::Mat source_gray;
    source = cv::imread("E:\\Code\\CalibrationTool\\data\\test3.jpg");
    cv::cvtColor(source, source_gray, cv::COLOR_RGB2GRAY);

    cv::Mat1d para;
    MtDualEllipseFitting::EllipseSolver ell(source_gray);
    int state = ell.compute(true);
    if (state == 0)
    {
        ell.getConicPara(para);

        cv::ellipse(source, cv::Point(para.at<double>(0, 0), para.at<double>(1, 0)),
            cv::Size(para.at<double>(2, 0), para.at<double>(3, 0)),
            para.at<double>(4, 0), 0, 360, cv::Scalar(0, 0, 255), 1, cv::LINE_8);

        imshow("result", source);
        cv::waitKey(0);
        cv::imwrite("E:\\Code\\EllipseFitSource\\result\\save3.jpg", source_gray);
    }
    else
        return;
    
}

void test(void)
{
    cv::Mat source;
    cv::Mat source_gray;
    source = cv::imread("E:\\Code\\EllipseFitSource\\data\\board.jpg");
    cv::cvtColor(source, source_gray, cv::COLOR_RGB2GRAY);

    RegionDetect detect;
    MatrixMat region;
    detect.setInputImage(source_gray);
    detect.compute(region,70);

    for (int i = 0; i < region.size(); i++)
    {
        cv::Mat1d para;
        MtDualEllipseFitting::EllipseSolver ell(region[i]);
        ell.compute(false);
        ell.getConicPara(para);

        cv::ellipse(source, cv::Point(para.at<double>(0, 0), para.at<double>(1, 0)),
            cv::Size(para.at<double>(2, 0), para.at<double>(3, 0)),
            para.at<double>(4, 0), 0, 360, cv::Scalar(0, 0, 255), 2, cv::LINE_8);
    }
    cv::imshow("result", source);
    cv::imwrite("E:\\Code\\EllipseFitSource\\result\\result.jpg", source);
    cv::waitKey(0);
}

void test_detect(void)
{
    cv::Mat source;
    cv::Mat source_gray;
    source = cv::imread("E:\\Code\\EllipseFitSource\\data\\board1.jpg");
    cv::cvtColor(source, source_gray, cv::COLOR_RGB2GRAY);

    RegionDetect detect;
    MatrixMat region;
    detect.setInputImage(source_gray);
    detect.compute(region,70);

    for (int i = 0; i < region.size(); i++)
    {
        std::string id = to_string(i);
        cv::imshow(id, region[i]);
        cv::imwrite("E:\\Code\\EllipseFitSource\\result\\save3" + to_string(i + 1) + ".jpg", region[i]);
    }
}

void testSpaceCircleSolver(void)
{
    /*std::string filename = "../data/circle.pcd";
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(filename, *pointcloud);*/
    
    float radiust = 2.6;
    Eigen::Vector3d O = Eigen::Vector3d::Zero();
    O(0) = 1.0;
    O(1) = 2.0;
    O(2) = 3.0;
    std::vector<Eigen::Vector3d> data;
    
   /* double std_x = 1.0;
    double std_y = 1.0;
    double std_z = 1.0;

    std::default_random_engine gen;*/
    
    for (int i = 0; i < 360; i += 5)
    {
        double theta = (i / 180.0) * PI;
        
        double x = O(0) + radiust * std::cos(theta);
        double y = O(1) + radiust * std::sin(theta);
        double z = O(2) + 0;

        /*normal_distribution<double> dist_x(x, std_x);
        normal_distribution<double> dist_y(y, std_y);
        normal_distribution<double> dist_z(z, std_z);*/

        Eigen::Vector3d point;
        /*point(0) = dist_x(gen);
        point(1) = dist_y(gen);
        point(2) = dist_z(gen);*/

        point(0) = x;
        point(1) = y;
        point(2) = z;

        data.push_back(point);
    }
    // add gross error
    Eigen::Vector3d point;
    point << 1, 2, 3;
    data.push_back(point);
    point << 2, 3, 5;
    data.push_back(point);

    double radius;
    Eigen::VectorXd center;
    SpaceCircleSolver solver(data, 0.02, true);
    solver.compute(center,radius);
    std::cout << "radius: " << radius << std::endl;
    std::cout << "center: " << center.transpose() << std::endl;
}

void testFeatureDetector(void)
{
    std::string files = "./data/14.bmp";
    
    std::vector<float> X;
    std::vector<float> Y;

    std::vector<cv::Point2d> corners;
    cv::Mat image = cv::imread(files);
    FeatureDetector detector(image,9,11);
    detector.compute(corners);

    cv::Size boardSize(9, 11);
    cv::drawChessboardCorners(image, boardSize, cv::Mat(corners), true);
    cv::namedWindow("corners", cv::WINDOW_NORMAL);
    cv::imshow("corners", image);
    cv::waitKey(0);
}

void testSpaceLineSolver(void)
{
    const double m = 1.0;
    const double n = 2.0;
    const double p = 3.0;

    std::vector<Eigen::Vector3d> data;
    
    for (int i = 0; i < 100; i++)
    {
        Eigen::Vector3d point;
        
        point(2) = i;
        point(1) = (n / p) * i;
        point(0) = (m / p) * i;

        data.push_back(point);
    }

    Eigen::Vector3d normal;
    SpaceLineSolver solver(data, 0.2, false);
    solver.compute(normal);
    
    std::cout << normal << std::endl;
}

void UndistortionKeys(const Eigen::Vector2d& vUnKeys, Eigen::Vector2d& vKeys, const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion)
{
    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);

    const double k1 = distortion(0);
    const double k2 = distortion(1);
    const double p1 = distortion(2);
    const double p2 = distortion(3);
    const double k3 = distortion(4);

    double u = vUnKeys(0);
    double v = vUnKeys(1);

    double xp = (u - cx) / fx;
    double yp = (v - cy) / fy;

    double r_2 = xp * xp + yp * yp;

    double xdis = xp * (double(1.) + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + double(2.) * p1 * xp * yp + p2 * (r_2 + double(2.) * xp * xp);
    double ydis = yp * (double(1.) + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + p1 * (r_2 + double(2.) * yp * yp) + double(2.) * p2 * xp * yp;

    double u_un = fx * xdis + cx;
    double v_un = fy * ydis + cy;

    vKeys = Eigen::Vector2d(u_un, v_un);

}

double CalculateScale(const Eigen::Vector2d& imagePoint,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3d& K,
    const Eigen::VectorXd& distortion,
    Eigen::Vector3d& x3D)
{
    double Zw = 0.0;
    Eigen::Vector2d unImagePoint;
    UndistortionKeys(imagePoint, unImagePoint, K, distortion);

    Eigen::Vector3d leftSideMatrix = Eigen::Vector3d::Zero();
    Eigen::Vector3d rightSideMatrix = Eigen::Vector3d::Zero();

    Eigen::Vector3d imageHomo = Eigen::Vector3d::Ones();
    imageHomo(0) = unImagePoint(0);
    imageHomo(1) = unImagePoint(1);

    leftSideMatrix = R.inverse() * K.inverse() * imageHomo;
    rightSideMatrix = R.inverse() * t;

    double scale = Zw + rightSideMatrix(2, 0) / leftSideMatrix(2, 0);

    //x3D = R.inverse() * (scale * K.inverse() * imageHomo - t);
    x3D = scale * K.inverse() * imageHomo;

    return scale;
}

double computeReprojectionErrors(const vector<Eigen::Vector3d>& objectPoints, const vector<vector<Eigen::Vector2d>>& imagePoints, const vector<Eigen::Matrix3d>& Rvecs,
    const vector<Eigen::Vector3d>& tvecs, const Eigen::Matrix3d& cameraMatrix, const Eigen::VectorXd& distCoeffs)
{
    int totalPoints = 0;
    double totalErr = 0.0;
    const int number = imagePoints.size();

    double k1 = distCoeffs(0), k2 = distCoeffs(1), k3 = distCoeffs(4);
    double p1 = distCoeffs(2), p2 = distCoeffs(3);
    double fx = cameraMatrix(0, 0), fy = cameraMatrix(1, 1);
    double cx = cameraMatrix(0, 2), cy = cameraMatrix(1, 2);

    for (int i = 0; i < number; i++)
    {
        Eigen::Matrix3d R = Rvecs[i];
        Eigen::Vector3d t = tvecs[i];
        const int n = imagePoints[i].size();
        double _errSum = 0.0;
        for (int j = 0; j < n; j++)
        {
            Eigen::Vector3d cam = R * objectPoints[j] + t;
            double xp = cam(0) / cam(2);
            double yp = cam(1) / cam(2);

            double u = fx * xp + cx;
            double v = fy * yp + cy;

            double _err = std::sqrt(pow(u - imagePoints[i][j](0), 2) + pow(v - imagePoints[i][j](1), 2));
            _errSum += _err;
        }
        std::cout << i << " reprojecction error: " << _errSum / n << std::endl;
        totalErr += _errSum;
        totalPoints += n;
    }
    return totalErr / totalPoints;
}


Matrix3d RotationVector2Matrix(const Eigen::Vector3d& v)
{
    double s = std::sqrt(v.dot(v));
    Eigen::Vector3d axis = v / s;
    Eigen::AngleAxisd r(s, axis);

    return r.toRotationMatrix();
}

Eigen::Vector3d RotationMatrix2Vector(const Eigen::Matrix3d& R)
{
    Eigen::AngleAxisd r;
    r.fromRotationMatrix(R);
    return r.angle() * r.axis();
}


void Optimize(const std::vector<Eigen::Vector3d>& objectPoints,
    const std::vector<Eigen::Vector2d>& imagePoints,
    Eigen::Matrix3d& R, Eigen::Vector3d& t, const Eigen::Matrix3d& K)
{

    Eigen::Vector3d r = RotationMatrix2Vector(R);

    {
        ceres::Problem problem;

        for (int i = 0; i < imagePoints.size(); ++i) {

            ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<ProjectCostRT, 2, 3, 3>(
                new ProjectCostRT(objectPoints[i], imagePoints[i], K));

            problem.AddResidualBlock(costFunction,
                nullptr,
                r.data(),
                t.data()
            );

        }
        std::cout << "Bundle Ajustment Solve Options ..." << std::endl;

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        options.preconditioner_type = ceres::JACOBI;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        options.function_tolerance = 1e-16;
        options.gradient_tolerance = 1e-16;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        R = RotationVector2Matrix(r);
    }
}

void testPnP(void)
{
    std::string strSettingFile = "E:\\Code\\EllipseFitSource\\config\\config.yaml";

    cv::FileStorage fs(strSettingFile.c_str(), cv::FileStorage::READ);

    double fx = fs["Camera_fx"];
    double fy = fs["Camera_fy"];
    double cx = fs["Camera_cx"];
    double cy = fs["Camera_cy"];

    double k1 = fs["Camera_k1"];
    double k2 = fs["Camera_k2"];
    double k3 = fs["Camera_k3"];
    double p1 = fs["Camera_p1"];
    double p2 = fs["Camera_p2"];
    
    Eigen::VectorXd distortion(5, 1); distortion << k1, k2, p1, p2, k3;
    Eigen::Matrix3d K; K << fx, 0, cx,0, fy, cy,0, 0, 1;
    
    std::vector<cv::Mat> RMat; fs["Image_Rotation"][0] >> RMat;
    std::vector<cv::Mat> tMat; fs["Image_Translation"][0] >> tMat;

    Eigen::Matrix3d R0; cv::cv2eigen(RMat[0], R0);
    Eigen::Vector3d t0; cv::cv2eigen(tMat[0], t0);
    Eigen::Matrix3d R1; cv::cv2eigen(RMat[1], R1);
    Eigen::Vector3d t1; cv::cv2eigen(tMat[1], t1);
    
    Eigen::Matrix4d T0 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();

    T0.topLeftCorner(3, 3) = R0; T0.topRightCorner(3, 1) = t0;
    T1.topLeftCorner(3, 3) = R1; T1.topRightCorner(3, 1) = t1;

    std::cout << "T0:" << T0 << std::endl;
    std::cout << "T1:" << T1 << std::endl;
   
    std::vector<std::string> files = {
      "E:\\Code\\EllipseFitSource\\data\\02\\Image_1.bmp",
      "E:\\Code\\EllipseFitSource\\data\\02\\Image_2.bmp",
    };

    cv::Mat img0 = cv::imread(files[0]);
    std::vector<cv::Point2d> corners0;
    FeatureDetector detect0(img0, 9, 11);
    detect0.compute(corners0);

    cv::Mat img1 = cv::imread(files[1]);
    std::vector<cv::Point2d> corners1;
    FeatureDetector detect1(img1, 9, 11);
    detect1.compute(corners1);

   /* cv::Size boardSize(9, 11);
    cv::drawChessboardCorners(img0, boardSize, cv::Mat(corners0), true);
    cv::drawChessboardCorners(img1, boardSize, cv::Mat(corners1), true);
    cv::namedWindow("corners0", cv::WINDOW_KEEPRATIO);
    cv::namedWindow("corners1", cv::WINDOW_KEEPRATIO);
    cv::imshow("corners0", img0);
    cv::imshow("corners1", img1);
    cv::waitKey(0);*/

    if (corners0.size() != corners1.size())
        return;
    
    std::vector<Eigen::Vector3d> pts3d;
    std::vector<Eigen::Vector2d> pts2d;
    std::vector<Eigen::Vector2d> pts2d0;
    std::vector<Eigen::Vector2d> pts2d1;

    for (int i = 0; i < corners0.size(); i++)
    {
        Eigen::Vector2d p_un,p(corners1[i].x, corners1[i].y);
        UndistortionKeys(p, p_un, K, distortion);
        pts2d.push_back(p_un);
        pts2d1.push_back(p_un);

        Eigen::Vector3d x3D;
        Eigen::Vector2d imagePoint_un,imagePoint(corners0[i].x, corners0[i].y);
        UndistortionKeys(imagePoint, imagePoint_un, K, distortion);
        pts2d0.push_back(imagePoint_un);
        CalculateScale(imagePoint, R0, t0, K, distortion, x3D);
        pts3d.push_back(x3D);
    }

    std::vector<std::vector<Eigen::Vector2d>> vImagePoints;
    vImagePoints.push_back(pts2d0);
    vImagePoints.push_back(pts2d1);

    Eigen::Matrix3d Rc;
    Eigen::Vector3d tc;
    MtPnPSolver::PnPSolver solver(pts3d,pts2d,strSettingFile);
    solver.SolvePnP(Rc, tc);

    Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
    Tc.topLeftCorner(3, 3) = Rc; Tc.topRightCorner(3, 1) = tc;

    Eigen::Matrix4d T10 = T1 * T0.inverse();
    
    std::cout << "T10: " << T10 << std::endl;
    std::cout << "Tc: " << Tc << std::endl;

    // compute reprojection error
    std::vector<Eigen::Matrix3d> vRList;
    std::vector<Eigen::Vector3d> vtList;
    
    Eigen::Matrix3d R_first = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_first = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_second = T10.topLeftCorner(3, 3);
    Eigen::Vector3d t_second = T10.topRightCorner(3, 1);

    vRList.push_back(R_first); vRList.push_back(R_second);
    vtList.push_back(t_first); vtList.push_back(t_second);

    double reprojectionError = computeReprojectionErrors(pts3d, vImagePoints, vRList, vtList, K, distortion);
    Optimize(pts3d, pts2d1, vRList[1], vtList[1], K);
    reprojectionError = computeReprojectionErrors(pts3d, vImagePoints, vRList, vtList, K, distortion);

    Tc.topLeftCorner(3, 3) = vRList[1]; Tc.topRightCorner(3, 1) = vtList[1];
    std::cout << "Tc: " << Tc << std::endl;
}

void testOptimize(const Eigen::MatrixXd& mat, const std::vector<Eigen::Vector3d>& world, const std::vector<Eigen::Vector2d>& pixel)
{
    {
        Eigen::VectorXd v(8);
        v(0) = mat(0, 0);
        v(1) = mat(0, 1);
        v(2) = mat(0, 2);
        v(3) = mat(0, 3);
        v(4) = mat(1, 0);
        v(5) = mat(1, 1);
        v(6) = mat(1, 2);
        v(7) = mat(1, 3);

        const int number = world.size();
        
        ceres::Problem optimizationProblem;
        for (int i = 0; i < number; i++)
        {
            optimizationProblem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<HomographyCost2, 1, 8>(new HomographyCost2(world[i](0), world[i](1), world[i](2), pixel[1](0),pixel[i](1))),
                nullptr,
                v.data()
            );
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &optimizationProblem, &summary);

        //std::cout << summary.BriefReport() << std::endl;
        std::cout << v << std::endl;
    }

    
}

void testZernikeMoment(void)
{
    std::string files = "E:\\Code\\EllipseFitSource\\data\\data.png";
    
    cv::Mat img_gray;
    cv::Mat img = cv::imread(files);
    cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);

    std::vector<float> x;
    std::vector<float> y;
    ZernikeMoment zm(img);
    zm.compute(x,y);
}

void testStereoCalibration(void)
{
    std::string Folder = "E:\\Code\\EllipseFitSource\\data\\calibration\\";
    
    std::vector<std::string> leftFiles =
    {
        Folder + "left\\left01.jpg",
        Folder + "left\\left02.jpg",
        Folder + "left\\left03.jpg",
        Folder + "left\\left04.jpg",
        Folder + "left\\left05.jpg",
        Folder + "left\\left06.jpg",
        Folder + "left\\left07.jpg",
        Folder + "left\\left08.jpg",
        Folder + "left\\left09.jpg",
        Folder + "left\\left11.jpg",
        Folder + "left\\left12.jpg",
        Folder + "left\\left13.jpg",
        Folder + "left\\left14.jpg",
    };

    std::vector<std::string> rightFiles =
    {
        Folder + "right\\right01.jpg",
        Folder + "right\\right02.jpg",
        Folder + "right\\right03.jpg",
        Folder + "right\\right04.jpg",
        Folder + "right\\right05.jpg",
        Folder + "right\\right06.jpg",
        Folder + "right\\right07.jpg",
        Folder + "right\\right08.jpg",
        Folder + "right\\right09.jpg",
        Folder + "right\\right11.jpg",
        Folder + "right\\right12.jpg",
        Folder + "right\\right13.jpg",
        Folder + "right\\right14.jpg",
    };

    // read data
    std::vector<cv::Mat> leftImages;
    std::vector<cv::Mat> rightImages;

    assert(leftFiles.size() == rightFiles.size());

    for (int it = 0; it < leftFiles.size(); it++)
    {
        leftImages.emplace_back(cv::imread(leftFiles[it]));
        rightImages.emplace_back(cv::imread(rightFiles[it]));
    }

    cv::Size boardSize(9, 6);
    cv::Size2f squareSize(25., 25.);

    std::string cameraParaPath = "E:\\Code\\EllipseFitSource\\config\\config_stereo.yaml";
    StereoCalibration calibration(4, leftFiles.size(), rightFiles.size(), cameraParaPath);
    calibration.SetLeftImages(leftImages);
    calibration.SetRightImages(rightImages);
    calibration.SetObjectPoints(boardSize, squareSize);
    calibration.compute();
    
    std::vector<cv::Mat> RList;
    std::vector<cv::Mat> tList;
    
    std::vector<std::vector<Eigen::Vector3d>> vLeftLandMark;
    std::vector<std::vector<Eigen::Vector3d>> vRightLandMark;
    std::vector<std::vector<Eigen::Vector3d>> vRightLandMarkMeasure;
    
    calibration.GetCameraPose(RList, tList);
    calibration.GetLeftLandMark(vLeftLandMark);
    calibration.GetRightLandMark(vRightLandMark);
    calibration.GetRightLandMarkMeasure(vRightLandMarkMeasure);

    pcl::PointCloud<pcl::PointXYZ>::Ptr leftCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr rightCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr rightCloudMeasure(new pcl::PointCloud<pcl::PointXYZ>);

    const int leftImageNum = vLeftLandMark.size();
    const int rightImageNum = vRightLandMark.size();
    const int leftFeatureNum = vLeftLandMark[0].size();
    const int rightFeatureNum = vRightLandMark[0].size();

    assert(leftImageNum == rightImageNum);
    assert(leftFeatureNum == rightFeatureNum);

    for (int i = 0; i < leftImageNum; i++)
    {
        for (int j = 0; j < leftFeatureNum; j++)
        {
            leftCloud->points.emplace_back(vLeftLandMark[i][j](0), vLeftLandMark[i][j](1), vLeftLandMark[i][j](2));
            rightCloud->points.emplace_back(vRightLandMark[i][j](0), vRightLandMark[i][j](1), vRightLandMark[i][j](2));
            rightCloudMeasure->points.emplace_back(vRightLandMarkMeasure[i][j](0), vRightLandMarkMeasure[i][j](1), vRightLandMarkMeasure[i][j](2));
        }
    }
    
    std::string fileLeft = "E:\\Code\\EllipseFitSource\\result\\leftcloud.pcd";
    std::string fileRight = "E:\\Code\\EllipseFitSource\\result\\rightcloud.pcd";
    std::string fileRightMeasure = "E:\\Code\\EllipseFitSource\\result\\rightcloudMeasure.pcd";

    leftCloud->height = 1; leftCloud->width = leftCloud->size();
    rightCloud->height = 1; rightCloud->width = rightCloud->size();
    rightCloudMeasure->height = 1; rightCloudMeasure->width = rightCloudMeasure->size();
    
    pcl::io::savePCDFileASCII(fileLeft, *leftCloud);
    pcl::io::savePCDFileASCII(fileRight, *rightCloud);
    pcl::io::savePCDFileASCII(fileRightMeasure, *rightCloudMeasure);

    // get transform matrix
    for (int it = 0; it < leftImageNum; it++)
    {
        std::cout << "R: " << std::endl;
        std::cout << RList[it] << std::endl;
        std::cout << "t: " << std::endl;
        std::cout << tList[it] << std::endl << std::endl;
        
    }
}

/*获取文件夹中所有文件的路径*/
void getFiles(const std::string& path, std::vector<std::string>& files)
{
	std::vector<cv::String> imagePaths;
	cv::glob(path, imagePaths, true);
	for (auto it : imagePaths) {
		files.push_back(it.c_str());
	}
	return;
}
/*构造二维2D特征点*/
void constructFeatures2D(const std::vector<std::vector<cv::Point2d>>& in,
	std::vector<std::vector<cv::Point2d>>& out)
{
	const int featureSize = in.begin()->size();
	//for (int it = 0; it < in.size();++it) {
	//	/*将所有点旋转90度*/
	//	std::vector<cv::Point2d> featurePointsR0;
	//	std::vector<cv::Point2d> featurePointsR90;
	//	featurePointsR0 = in[it];
	//	featurePointsR90.resize(featureSize);
	//	for (int iti = 0; iti < featureSize; ++iti) {
	//		cv::Point2d refPt = in[it][0];
	//		CommonFunctions::RotatePoint(featurePointsR0[iti], refPt, 0.5*PI, featurePointsR90[iti]);
	//	}
	//	/*拟合直线*/
	//	CommonStruct::LineFunction2D lineR0 = CommonFunctions::ComputeLineFunction2D(featurePointsR0);
	//	CommonStruct::LineFunction2D lineR90 = CommonFunctions::ComputeLineFunction2D(featurePointsR90);
	//	/*计算特征*/
	//	std::vector<cv::Point2d> featurePoints;
	//	for (int iti = 0; iti < featureSize; ++iti) {
	//		for (int itj = 0; itj < featureSize; ++itj) {
	//			CommonStruct::LineFunction2D lineRow = CommonFunctions::ComputeLineFunction2D(lineR0, featurePointsR90[iti]);
	//			CommonStruct::LineFunction2D lineCol = CommonFunctions::ComputeLineFunction2D(lineR90, featurePointsR0[itj]);
	//			featurePoints.push_back(CommonFunctions::ComputeIntersectionPt(lineRow, lineCol));
	//		}
	//	}
	//	out.push_back(featurePoints);
	//}
	for (int it = 0; it < in.size();++it) {
		/*计算特征*/
		std::vector<cv::Point2d> featurePoints;
		for (int iti = 0; iti < featureSize; ++iti) {
			for (int itj = 0; itj < featureSize; ++itj) {
				featurePoints.push_back(cv::Point2f(in[it][iti].x, in[it][itj].x));
			}
		}
		out.push_back(featurePoints);
	}
	return;
}
/*构造三维3D特征点*/
void constructFeatures3D(const std::vector<std::vector<cv::Point3d>>& in,
	std::vector<std::vector<cv::Point3d>>& out)
{
	/*将3D数据去除Z分量转化为2D数据*/
	std::vector<std::vector<cv::Point2d>> in2Ds;
	std::vector<std::vector<cv::Point2d>> out2Ds;
	for (int iti = 0; iti < in.size(); ++iti) {
		std::vector<cv::Point2d> in2D;
		for (int itj = 0; itj < in[iti].size(); ++itj) {
			in2D.push_back(cv::Point2d(in[iti][itj].x, in[iti][itj].y));
		}
		in2Ds.push_back(in2D);
	}
	const int featureSize = in.begin()->size();
	for (int it = 0; it < in.size();++it) {
		/*将所有点旋转90度*/
		std::vector<cv::Point2d> featurePointsR0;
		std::vector<cv::Point2d> featurePointsR90;
		featurePointsR0 = in2Ds[it];
		featurePointsR90.resize(featureSize);
		for (int iti = 0; iti < featureSize; ++iti) {
			cv::Point2d refPt = in2Ds[it][0];
			CommonFunctions::RotatePoint(featurePointsR0[iti], refPt, 0.5*PI, featurePointsR90[iti]);
		}
		/*拟合直线*/
		CommonStruct::LineFunction2D lineR0 = CommonFunctions::ComputeLineFunction2D(featurePointsR0);
		CommonStruct::LineFunction2D lineR90 = CommonFunctions::ComputeLineFunction2D(featurePointsR90);
		/*计算特征*/
		std::vector<cv::Point2d> featurePoints;
		for (int iti = 0; iti < featureSize; ++iti) {
			for (int itj = 0; itj < featureSize; ++itj) {
				CommonStruct::LineFunction2D lineRow = CommonFunctions::ComputeLineFunction2D(lineR0, featurePointsR90[iti]);
				CommonStruct::LineFunction2D lineCol = CommonFunctions::ComputeLineFunction2D(lineR90, featurePointsR0[itj]);
				featurePoints.push_back(CommonFunctions::ComputeIntersectionPt(lineRow, lineCol));
			}
		}
		out2Ds.push_back(featurePoints);
	}
	for (int iti = 0; iti < out2Ds.size(); ++iti) {
		std::vector<cv::Point3d> out3D;
		for (int itj = 0; itj < out2Ds[iti].size(); ++itj) {
			out3D.push_back(cv::Point3d(out2Ds[iti][itj].x, out2Ds[iti][itj].y, 0.0));
		}
		out.push_back(out3D);
	}
	return;
}

int main()
{
	//RectifyingImage();
    //testZernikeMoment();
    //testPnP();
    //testSpaceLineSolver();
    //testFeatureDetector();
    //test3DConstruct();
    //testSpaceCircleSolver();
    //generate_ellipse();
    //test();
    //test2();
    //testHomography();
    //testCalibration();
    //testDualEllipse();
    //test_detect();
    //test_point_ellipse();
    //test_ellipse_fit();
    //testStereoCalibration();
	/*提取线扫相机特征点*/
	const int featuresNum = 31;
	std::string left_dir = "C:\\Users\\Administrator\\Desktop\\LineScanCaliData\\left";
	std::string right_fir = "C:\\Users\\Administrator\\Desktop\\LineScanCaliData\\right";
	std::vector<std::string> left_files;
	std::vector<std::string> right_files;
	getFiles(left_dir.c_str(), left_files);
	getFiles(right_fir.c_str(), right_files);
	//初始化特征参数
	const int imageNum = left_files.size();
	assert(left_files.size() == right_files.size());
	//左相机标定
	std::vector<std::vector<cv::Point2d>> test2Dset;
	std::vector<std::vector<cv::Point3d>> test3Dset;
	//提取2D&3D特征
	for (int it = 0; it < imageNum; ++it) {
		//确定图像高度
		float imageHeight = 0.;
		int ops = left_files[it].rfind('.');
		if (left_files[it][ops - 4] == '-') {
			std::string t_h = left_files[it].substr(ops - 4, 4);
			const char *p = t_h.data();
			imageHeight = atof(p);
			imageHeight /= -10.;
			imageHeight += 1000.0;
		}
		else {
			std::string t_h = left_files[it].substr(ops - 3, 3);
			const char *p = t_h.data();
			imageHeight = atof(p);
			imageHeight /= -10.;
			imageHeight += 1000.0;
		}
		//读取图像
		cv::Mat imageRaw = cv::imread(left_files[it], cv::IMREAD_GRAYSCALE);
		/*cv::flip(imageRaw, imageRaw, 1);
		cv::imwrite("C:\\Users\\Administrator\\Desktop\\flip.bmp", imageRaw);*/
		if (imageRaw.empty()) {
			//检查是否读取图像
			std::cout << "Error! Input image cannot be read...\n";
			return -1;
		}
		//提取图像特征
		FeaturesPointExtract featuresEx(featuresNum, imageHeight);
		unsigned char *pImage = imageRaw.data;
		featuresEx.SetFeatureImage(pImage, 16384, 1600);
		featuresEx.SetCalibrationPara(1000.0, 4000.0); //10um/pixel
		featuresEx.Update();
		std::vector<cv::Point2d> test2D;
		std::vector<cv::Point3d> test3D;
		featuresEx.Get2DPoints(test2D);
		featuresEx.Get3DPoints(test3D);
		test2Dset.push_back(test2D);
		test3Dset.push_back(test3D);
	}
	/*由一维构造二维特征点*/
	CommonFunctions::ExchageXY(test2Dset);
	std::vector<std::vector<cv::Point2d>> TwoDFeaturePoints;
	std::vector<std::vector<cv::Point3d>> ThreeDFeaturePoints;
	constructFeatures2D(test2Dset, TwoDFeaturePoints);
	constructFeatures3D(test3Dset, ThreeDFeaturePoints);
	/*加入张正友标定中获取内外参信息*/
	const int distortionParameterNum = 4;
	const std::string cameraParaPath = "E:\\Code\\EllipseFitSource\\config\\config.yaml";
	ZZYCalibration calibration(distortionParameterNum, imageNum, cameraParaPath);
	calibration.setObjectPoints(ThreeDFeaturePoints);
	calibration.setImagePoints(TwoDFeaturePoints);
	Eigen::Matrix3d CameraMatrix;
	Eigen::Vector3d RadialDistortion;
	Eigen::Vector2d TangentialDistortion;
	std::vector<std::vector<Eigen::Vector3d>> vX3D;
	calibration.compute(CameraMatrix, RadialDistortion, TangentialDistortion);
	calibration.get3DPoint(vX3D);
	//std::vector<std::vector<cv::Point2d>> test2Dset_r;
	//std::vector<std::vector<cv::Point3d>> test3Dset_r;
	//for (int it = 0; it < 1; ++it) {
	//	//确定图像高度
	//	float imageHeight = 0.;
	//	int ops = right_files[it].rfind('.');
	//	if (right_files[it][ops - 4] == '-') {
	//		std::string t_h = right_files[it].substr(ops - 4, 4);
	//		const char *p = t_h.data();
	//		imageHeight = atof(p);
	//		imageHeight /= 10.;
	//		imageHeight += 1000.0;
	//	}
	//	else {
	//		std::string t_h = right_files[it].substr(ops - 3, 3);
	//		const char *p = t_h.data();
	//		imageHeight = atof(p);
	//		imageHeight /= 10.;
	//		imageHeight += 1000.0;
	//	}
	//	//读取图像
	//	cv::Mat imageRaw = cv::imread(right_files[it],cv::IMREAD_GRAYSCALE);
	//	/*cv::flip(imageRaw, imageRaw, 0);*/
	//	if (imageRaw.empty()) {
	//		//检查是否读取图像
	//		std::cout << "Error! Input image cannot be read...\n";
	//		return -1;
	//	}
	//	//提取图像特征
	//	FeaturesPointExtract featuresEx(featuresNum, imageHeight);
	//	unsigned char *pImage = imageRaw.data;
	//	featuresEx.SetFeatureImage(pImage, 16384, 1600);
	//	featuresEx.SetCalibrationPara(1000.0, 4000.0); //10um/pixel
	//	featuresEx.Update();
	//	std::vector<cv::Point2d> test2D;
	//	std::vector<cv::Point3d> test3D;
	//	featuresEx.Get2DPoints(test2D);
	//	featuresEx.Get3DPoints(test3D);
	//	test2Dset_r.push_back(test2D);
	//	test3Dset_r.push_back(test3D);
	//}
	////右相机标定
	//LineScanCalibration LC_r;
	//LineScanPara P_r;
	//std::vector<LineScanPara> initP_r;
	//LC_r.SetImagePoints(test2Dset_r);
	//LC_r.SetObjectPoints(test3Dset_r);
	//LC_r.Update();
	//LC_r.GetCameraPara(P_r);
	//LC_r.GetInitCameraPara(initP_r);
	//for (int it = 0; it < 1; ++it) {
	//	std::cout << "Right " << "第" << it << "帧" << std::endl;
	//	std::cout << "R:" << initP_r[it].R << std::endl;
	//	std::cout << "R det:" << initP_r[it].R.determinant() << std::endl;
	//	std::cout << "T:" << initP_r[it].T << std::endl;
	//	std::cout << "k1:" << initP_r[it].k1 << std::endl;
	//	std::cout << "k2:" << initP_r[it].k2 << std::endl;
	//	std::cout << "k3:" << initP_r[it].k3 << std::endl;
	//	std::cout << "vc:" << initP_r[it].vc << std::endl;
	//	std::cout << "Fy:" << initP_r[it].Fy << std::endl;
	//	std::cout << "resY:" << initP_r[it].resY << std::endl;
	//	std::cout << "Conf:" << initP_r[it].Conf << std::endl;
	//}
	//std::cout << "Right Result:" << std::endl;
	//std::cout << "R:" << P_r.RList[0] << std::endl;
	//std::cout << "R det:" << P_r.RList[0].determinant() << std::endl;
	//std::cout << "T:" << P_r.TList[0] << std::endl;
	//std::cout << "k1:" << P_r.k1 << std::endl;
	//std::cout << "k2:" << P_r.k2 << std::endl;
	//std::cout << "k3:" << P_r.k3 << std::endl;
	//std::cout << "vc:" << P_r.vc << std::endl;
	//std::cout << "Fy:" << P_r.Fy << std::endl;
	//std::cout << "resY:" << P_r.resY << std::endl;
	//std::cout << "Conf:" << P_r.Conf << std::endl;
	system("pause");
	return 0;
}