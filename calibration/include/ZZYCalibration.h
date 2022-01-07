#pragma once
#include<Eigen/Core>
#include<Eigen/SVD>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/calib3d.hpp>
#include<iostream>
#include"CeresCostFun.h"

using namespace std;
using namespace Eigen;

namespace MtZZYCalibration
{
	struct TwoDCoordinate
	{
		double x;
		double y;
		TwoDCoordinate() :x(0.0), y(0.0) {};
	};

	struct ThreeDCoordinate
	{
		double x;
		double y;
		double z;
		ThreeDCoordinate() :x(0.0), y(0.0), z(0.0) {};
	};

	class ZZYCalibrationFailure
	{
	public:
		ZZYCalibrationFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class ZZYCalibration
	{
	public:

		ZZYCalibration(const int& DistortionParaNum, const int& ImageNum, const std::string& cameraParaPath) 
			:mDistortionParaNum(DistortionParaNum),mImageNum(ImageNum),mStrCameraParaPath(cameraParaPath) {};

		void setObjectPoints(const std::vector<std::vector<Eigen::Vector3d>> object) { mObjectPoints = object; }
		void setImagePoints(const std::vector<std::vector<Eigen::Vector2d>> image) { mImagePoints = image; }
		void compute(Eigen::Matrix3d& CameraMatrix, Eigen::Vector3d& RadialDistortion, Eigen::Vector2d& TangentialDistortion);
		
		void get3DPoint(std::vector<std::vector<Eigen::Vector3d>>& vX3D) { vX3D = mvX3D; };
		void getCameraPose(std::vector<cv::Mat>& RList, std::vector<cv::Mat>& tList) { RList = mRListMat; tList = mtListMat; };

		void getObjectPoints1(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<Eigen::Vector3d>& objectPoints);
		void getObjectPoints2(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<Eigen::Vector3d>& objectPoints);
		bool findHomography(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H, bool isNormal = false)throw(ZZYCalibrationFailure);
		bool findHomographyByRansac(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H, bool isNormal = false)throw(ZZYCalibrationFailure);
		bool findHomographyByOpenCV(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H)throw(ZZYCalibrationFailure);
	private:
		const int mImageNum;
		std::string mStrCameraParaPath;
		std::vector<std::vector<Eigen::Vector3d>> mObjectPoints;
		std::vector<std::vector<Eigen::Vector2d>> mImagePoints;
		
		const int mDistortionParaNum;
		Eigen::Matrix3d mCameraMatrix;
		Eigen::Vector3d mRadialDistortion;
		Eigen::Vector2d mTangentialDistortion;

		std::vector<cv::Mat> mRListMat;
		std::vector<cv::Mat> mtListMat;

		std::vector<std::vector<Eigen::Vector3d>> mvX3D;
		// method
		bool Normalize(Eigen::MatrixXd& P, Eigen::Matrix3d& T);
		bool Normalize(const std::vector<Eigen::Vector2d>& vKeys, std::vector<Eigen::Vector2d>& vNormalizedPoints, Eigen::Matrix3d& T);

		int rand_int(void) { return std::rand(); }
		Eigen::VectorXd constructVector(const Matrix3d& H, int i, int j);
		void PrintCameraIntrinsics();

		Matrix3d RotationVector2Matrix(const Eigen::Vector3d& v);
		Eigen::Vector3d RotationMatrix2Vector(const Eigen::Matrix3d& R);

		Eigen::VectorXd solveHomographyDLT(const Eigen::MatrixXd& srcPoints, const Eigen::MatrixXd& dstPoints)throw(ZZYCalibrationFailure);
		Eigen::Matrix3d solveInitCameraIntrinstic(const std::vector<Eigen::Matrix3d>& homography);
		void solveInitCameraExtrinstic(const std::vector<Matrix3d>& homographies, const Eigen::Matrix3d& K, std::vector<Eigen::Matrix3d>& RList, std::vector<Vector3d>& tList);
		
		void computeCameraCalibration(std::vector<std::vector<Eigen::Vector2d>>& imagePoints,
			std::vector<std::vector<Eigen::Vector3d>>& objectPoints,
			cv::Mat& cameraMatrix, cv::Mat& distCoeffs)throw(ZZYCalibrationFailure);
		double computeReprojectionErrors(const vector<vector<Eigen::Vector3d>>& objectPoints, const vector<vector<Eigen::Vector2d>>& imagePoints, const vector<Eigen::Vector3d>& rvecs, 
			const vector<Eigen::Vector3d>& tvecs, const Eigen::Matrix3d& cameraMatrix, const Eigen::VectorXd& distCoeffs, vector<double>& perViewErrors)throw(ZZYCalibrationFailure);

		// 3d construct
		double CalculateScale(const Eigen::Vector2d& imagePoint, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion,Eigen::Vector3d& x3D);
		void UndistortionKeys(const Eigen::Vector2d& vUnKeys, Eigen::Vector2d& vKeys, const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion);
		void Optimize(const Eigen::Matrix3d& K, std::vector<Eigen::Matrix3d>& RList, std::vector<Eigen::Vector3d>& tList, std::vector<std::vector<Eigen::Vector3d>>& vP3D);
		void Triangulate(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2, const Eigen::MatrixXd& P1, const Eigen::MatrixXd& P2, Eigen::Vector3d& x3D);
		void DepthRecover(const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion, const std::vector<Eigen::Matrix3d>& RList, const std::vector<Eigen::Vector3d>& tList, std::vector<std::vector<Eigen::Vector3d>>& vX3D);
	};
}