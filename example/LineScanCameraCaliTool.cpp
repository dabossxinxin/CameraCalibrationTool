/*线扫相机标定程序*/
#include "CommonFunctions.h"
#include "LineScanCalibration.h"
#include "GrowAllSpecifiedRegions.h"
#include <stdio.h>
#include <vector>
#include <regex>
#include <algorithm>
#include <stdio.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <fstream>

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
/*保存世界点坐标*/
void saveWorldPoints(std::string& filename, std::vector<cv::Point3d>& pts)
{
	std::ofstream outfile;
	outfile.open(filename, std::ofstream::app);
	for (int it = 0; it < pts.size(); ++it) {
		outfile << pts[it].x << "\t" << pts[it].y << "\t" << pts[it].z << "\t" << std::endl;
	}
	outfile.close();
}
/*计算去畸变图像*/
void SaveUndistortImage(const cv::Mat& inputImage,
	const LineScanPara& cameraPara,
	cv::Mat& outputImage)
{
	/*获取畸变参数*/
	const double k1 = cameraPara.k1;
	const double k2 = cameraPara.k2;
	const double k3 = cameraPara.k3;
	const double fy = cameraPara.Fy;
	const double vc = cameraPara.vc;
	Eigen::Matrix3d K =
		Eigen::Matrix3d::Zero();
	K(0, 0) = K(2, 2) = 1.0;
	K(1, 1) = fy;
	K(1, 2) = vc;
	/*获取图像信息*/
	const int imageRow = inputImage.rows;
	const int imageCol = inputImage.cols;
	/*初始化去畸变图像*/
	outputImage = cv::Mat(imageRow, imageCol, CV_8UC1, cv::Scalar(0));
	for (int row = 0; row < imageRow; ++row) {
		for (int col = 0; col < imageCol; ++col) {
			double u = col;
			double v = row;
			double u_n = (u - vc) / fy;
			u_n = u_n - (k1*u_n*u_n*u_n + k2*u_n*u_n*u_n*u_n*u_n + k3*u_n*u_n);
			u = fy*u_n + vc;
			/*最邻近插值*/
			if (u >= 0 && u < imageCol&&v >= 0 && v < imageRow) {
				outputImage.at<uchar>(row, col) = inputImage.at<uchar>(int(v), int(u));
			}
		}
	}
}
/*提取图像2D&3D特征*/
void ExtractFeaturePoints(const std::vector<std::string>& imageFiles,
	const std::string& debugPath,
	std::vector<std::vector<cv::Point2d>>& features2D,
	std::vector<std::vector<cv::Point3d>>& features3D)
{
	/*初始化特征参数*/
	features2D.clear();
	features3D.clear();
	const int imageNum = imageFiles.size();
	for (int it = 5; it < imageNum; it += 1) {
		/*确定图像高度*/
		float imageHeight = 0.;
		int ops = imageFiles[it].rfind('.');
		if (imageFiles[it][ops - 4] == '-') {
			std::string t_h = imageFiles[it].substr(ops - 4, 4);
			const char *p = t_h.data();
			imageHeight = atof(p);
			imageHeight /= -1000.;//mm
			imageHeight += 0.0;
		}
		else {
			std::string t_h = imageFiles[it].substr(ops - 3, 3);
			const char *p = t_h.data();
			imageHeight = atof(p);
			imageHeight /= -1000.;//mm
			imageHeight += 0.0;
		}
		/*读取图像*/
		cv::Mat imageRaw = cv::imread(imageFiles[it], cv::IMREAD_GRAYSCALE);
		if (imageRaw.empty()) {
			//检查是否读取图像
			std::cout << "Error! Input image cannot be read...\n";
			return;
		}
		/*提取图像特征*/
		FeaturesPointExtract featuresEx(31, imageHeight);
		unsigned char *pImage = imageRaw.data;
		featuresEx.SetFeatureImage(pImage, 16384, 1600);
		featuresEx.SetDebugPath(debugPath);
		featuresEx.SetCalibrationPara(10.0, 40.0); //10um/pixel
		featuresEx.Update();
		std::vector<cv::Point2d> test2D;
		std::vector<cv::Point3d> test3D;
		featuresEx.Get2DPoints(test2D);
		featuresEx.Get3DPoints(test3D);
		features2D.push_back(test2D);
		features3D.push_back(test3D);
	}
}
/*相机标定*/
void CalibrationLineScanCamera(const std::string flag,
	const std::string debugPath,
	std::vector<std::vector<cv::Point2d>>& features2D,
	std::vector<std::vector<cv::Point3d>>& features3D,
	LineScanPara& cameraPara)
{
	LineScanCalibration calibration;
	std::vector<LineScanPara> cameraInitPara;
	std::vector<cv::Point3d> groundTruth;
	std::vector<cv::Point3d> worldPtsBeforeOptimized;
	std::vector<cv::Point3d> worldPtsAfterOptimized;
	calibration.SetImagePoints(features2D);
	calibration.SetObjectPoints(features3D);
	calibration.SetDebugPath(debugPath);
	calibration.Update();
	calibration.GetCameraPara(cameraPara);
	calibration.GetInitCameraPara(cameraInitPara);
	calibration.GetWorldPointsBeforeOptimized(worldPtsBeforeOptimized);
	calibration.GetWorldPointsAfterOptimized(worldPtsAfterOptimized);
	calibration.GetGroundTruth(groundTruth);
	/*保存世界坐标3D点*/
	std::string saveWorldPtsPathBO = debugPath + "\\WorldBeforeOptimized" + flag + ".txt";
	std::string saveWorldPtsPathAO = debugPath + "\\WorldAfterOptimized" + flag + ".txt";
	std::string saveGroundTruthPtsPath = debugPath + "\\GroundTruth" + flag + ".txt";
	saveWorldPoints(saveWorldPtsPathBO, worldPtsBeforeOptimized);
	saveWorldPoints(saveWorldPtsPathAO, worldPtsAfterOptimized);
	saveWorldPoints(saveGroundTruthPtsPath, groundTruth);
	/*打印相机标定信息*/
	std::cout << "Init Calibration Info " << flag << std::endl;
	for (int it = 0; it < cameraInitPara.size(); ++it) {
		std::cout << "\tR:" << cameraInitPara[it].R << std::endl;
		std::cout << "\tR det:" << cameraInitPara[it].R.determinant() << std::endl;
		std::cout << "\tT:" << cameraInitPara[it].T << std::endl;
		std::cout << "\tk1:" << cameraInitPara[it].k1 << std::endl;
		std::cout << "\tk2:" << cameraInitPara[it].k2 << std::endl;
		std::cout << "\tk3:" << cameraInitPara[it].k3 << std::endl;
		std::cout << "\tvc:" << cameraInitPara[it].vc << std::endl;
		std::cout << "\tFy:" << cameraInitPara[it].Fy << std::endl;
		//std::cout << "\tresY:" << cameraInitPara[it].resY << std::endl;
		//std::cout << "\tConf:" << cameraInitPara[it].Conf << std::endl;
	}
	std::cout << "Optimized Calibration Info " << flag << std::endl;
	std::cout << "\tR:" << cameraPara.R << std::endl;
	std::cout << "\tR det:" << cameraPara.R.determinant() << std::endl;
	std::cout << "\tT:" << cameraPara.T << std::endl;
	std::cout << "\tk1:" << cameraPara.k1 << std::endl;
	std::cout << "\tk2:" << cameraPara.k2 << std::endl;
	std::cout << "\tk3:" << cameraPara.k3 << std::endl;
	std::cout << "\tvc:" << cameraPara.vc << std::endl;
	std::cout << "\tFy:" << cameraPara.Fy << std::endl;
	std::cout << "\tresY:" << cameraPara.resY << std::endl;
	std::cout << "\tConf:" << (cameraPara.Conf == true ? "OK" : "NG") << std::endl;
}

int main()
{
	/*设置工作目录*/
	std::string left_dir = "C:\\Users\\Administrator\\Desktop\\LineScanCaliData\\left";
	std::string right_fir = "C:\\Users\\Administrator\\Desktop\\LineScanCaliData\\right";
	/*获取文件路径*/
	std::vector<std::string> left_files;
	std::vector<std::string> right_files;
	getFiles(left_dir.c_str(), left_files);
	getFiles(right_fir.c_str(), right_files);
	/*初始化特征参数*/
	LineScanPara leftCameraPara;
	LineScanPara rightCameraPara;
	std::string leftCamera = "Left";
	std::string rightCamera = "Right";
	std::string debugPath = "C:\\Users\\Administrator\\Desktop\\DebugInfo";
	assert(left_files.size() == right_files.size());
	/*删除debugPath文件夹中积压的文件*/
	
	/*左相机标定*/
	std::vector<std::vector<cv::Point2d>> features2DLeft;
	std::vector<std::vector<cv::Point3d>> features3DLeft;
	/*提取2D&3D特征*/
	ExtractFeaturePoints(left_files, debugPath, features2DLeft, features3DLeft);
	CalibrationLineScanCamera(leftCamera, debugPath, features2DLeft, features3DLeft, leftCameraPara);	
	/*去除图像畸变*/
	cv::Mat UndistortImageLeft;
	cv::Mat leftImage = cv::imread(left_files[0], cv::IMREAD_GRAYSCALE);
	SaveUndistortImage(leftImage, leftCameraPara, UndistortImageLeft);
	std::string undistortImageFile = debugPath+"\\UndistortLeft.bmp";
	cv::imwrite(undistortImageFile, UndistortImageLeft);
	/*右相机标定*/
	std::vector<std::vector<cv::Point2d>> features2DRight;
	std::vector<std::vector<cv::Point3d>> features3DRight;
	/*提取2D&3D特征*/
	ExtractFeaturePoints(right_files, debugPath, features2DRight, features3DRight);
	CalibrationLineScanCamera(rightCamera, debugPath, features2DRight, features3DRight, rightCameraPara);
	/*去除图像畸变*/
	cv::Mat UndistortImageRight;
	cv::Mat rightImage = cv::imread(right_files[0], cv::IMREAD_GRAYSCALE);
	SaveUndistortImage(rightImage, rightCameraPara, UndistortImageRight);
	std::string undistortRightImageFile = debugPath+"\\UndistortRight.bmp";
	cv::imwrite(undistortRightImageFile, UndistortImageRight);
	/*正常返回*/
	system("pause");
	return 0;
}