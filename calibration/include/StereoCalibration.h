#pragma once
#include<Eigen/Core>
#include<Eigen/SVD>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/calib3d.hpp>
#include<iostream>
#include"CeresCostFun.h"
#include"Random.h"
#include"ZZYCalibration.h"
#include"FeatureDetector.h"

using namespace std;
using namespace Eigen;
using namespace MtZZYCalibration;
using namespace MtFeatureDetector;


namespace MtStereoCalibration
{

	class StereoCalibrationFailure
	{

	public:
		StereoCalibrationFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class StereoCalibration
	{
	public:
		StereoCalibration(const int& paraNum, const int& leftImageNum, const int& rightImageNum, const std::string& cameraParaPath)
			:mDistortionParaNum(paraNum), mLeftImageNum(leftImageNum), 
			mRightImageNum(rightImageNum), mCameraParaPath(cameraParaPath)
		{
			assert(mLeftImageNum == mRightImageNum);
			mImageNum = mLeftImageNum;
		};

		void SetLeftImages(const std::vector<cv::Mat>& leftImages)
		{
			mLeftImage = leftImages;
		}
		void SetRightImages(const std::vector<cv::Mat>& rightImages)
		{
			mRightImage = rightImages;
		}
		void SetObjectPoints(const cv::Size& borderSize, const cv::Size2f& squareSize)
		{
			mBorderSize = borderSize;
			mSquareSize = squareSize;
			//GenerateObjectPointsOwner(mBorderSize, mSquareSize, mObjectPoints);
			GenerateObjectPointsCV(mBorderSize, mSquareSize, mObjectPoints);
		}
			
		void compute();

		void GetCameraPose(std::vector<cv::Mat>& R, std::vector<cv::Mat>& t) { R = mRListMat, t = mtListMat; };
		void GetLeftLandMark(std::vector<std::vector<Eigen::Vector3d>>& landMark) { landMark = mLeftLandMark; };
		void GetRightLandMark(std::vector<std::vector<Eigen::Vector3d>>& landMark) { landMark = mRightLandMark; };
		void GetRightLandMarkMeasure(std::vector<std::vector<Eigen::Vector3d>>& landMark) { landMark = mRightLandMarkMeasure; }
	
	private:
		std::vector<cv::Mat> mLeftImage;
		std::vector<cv::Mat> mRightImage;

		std::vector<std::vector<Eigen::Vector2d>> mLeftImagePoints;
		std::vector<std::vector<Eigen::Vector2d>> mRightImagePoints;
		std::vector<std::vector<Eigen::Vector3d>> mObjectPoints;

		int mImageNum;
		cv::Size mBorderSize;
		cv::Size2f mSquareSize;
		const int mLeftImageNum;
		const int mRightImageNum;
		const int mDistortionParaNum;
		const std::string mCameraParaPath;

		Eigen::Matrix3d mLeftCameraMatrix;
		Eigen::Vector3d mLeftRadialDistortion;
		Eigen::Vector2d mLeftTangentialDistortion;

		Eigen::Matrix3d mRightCameraMatrix;
		Eigen::Vector3d mRightRadialDistortion;
		Eigen::Vector2d mRightTangentialDistortion;

		std::vector<cv::Mat> mLeftRListMat;
		std::vector<cv::Mat> mLefttListMat;
		
		std::vector<cv::Mat> mRightRListMat;
		std::vector<cv::Mat> mRighttListMat;

		std::vector<cv::Mat> mRListMat;
		std::vector<cv::Mat> mtListMat;

		cv::Mat mRExtrinsic;
		cv::Mat mtExtrinsic;

		std::vector<std::vector<Eigen::Vector3d>> mLeftLandMark;
		std::vector<std::vector<Eigen::Vector3d>> mRightLandMark;
		std::vector<std::vector<Eigen::Vector3d>> mRightLandMarkMeasure;

		// method
		void GenerateObjectPointsOwner(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<std::vector<Eigen::Vector3d>>& objectPoints);
		void GenerateObjectPointsCV(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<std::vector<Eigen::Vector3d>>& objectPoints);

		void GenerateImagePointsOwner(const std::vector<cv::Mat>& images, const std::string& id, std::vector<std::vector<Eigen::Vector2d>>& imagePoints);
		void GenerateImagePointsCV(const std::vector<cv::Mat>& images, const std::string& id, std::vector<std::vector<Eigen::Vector2d>>& imagePoints);

		void MonocularCalibration(const std::string& id);
		void ComputeReprojectionError(const std::vector<cv::Mat>& R, const std::vector<cv::Mat>& t);

		void ComputeRectificationParameter(const cv::Mat& R, const cv::Mat& t, Eigen::Matrix3d& RLeft, Eigen::Matrix3d& RRight);
		void Rectification(const cv::Mat& leftImage, const cv::Mat& rightImage,
						   const Eigen::Matrix3d& RLeft, const Eigen::Matrix3d& RRight,
						   cv::Mat& leftImageRectification, cv::Mat& rightImageRectification);
		
	};
}