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

typedef std::vector<std::vector<cv::Point2d>> doublePoint2D;
typedef std::vector<std::vector<cv::Point3d>> doublePoint3D;
typedef std::vector<std::vector<Eigen::Vector2d>> doubleVector2D;
typedef std::vector<std::vector<Eigen::Vector3d>> doubleVector3D;

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
		/*���캯��*/
		ZZYCalibration(const int& DistortionParaNum, const int& ImageNum, const std::string& cameraParaPath) 
			:mDistortionParaNum(DistortionParaNum),mImageNum(ImageNum),mStrCameraParaPath(cameraParaPath) {};
		/*�������ú���*/
		void setObjectPoints(const std::vector<std::vector<cv::Point3d>>& object) {
			this->Points3d2Vectors3d(object, this->mObjectPoints);
		}
		void setImagePoints(const std::vector<std::vector<cv::Point2d>>& image) {
			this->Points2d2Vectors2d(image, this->mImagePoints);
		}
		void setObjectPoints(const std::vector<std::vector<Eigen::Vector3d>> object) { 
			mObjectPoints = object; 
		}
		void setImagePoints(const std::vector<std::vector<Eigen::Vector2d>> image) { 
			mImagePoints = image; 
		}
		/*�궨��������*/
		void compute(Eigen::Matrix3d& CameraMatrix, Eigen::Vector3d& RadialDistortion, Eigen::Vector2d& TangentialDistortion);
		/*��ȡ�궨��Ϣ*/
		void get3DPoint(std::vector<std::vector<Eigen::Vector3d>>& vX3D) { 
			vX3D = mvX3D; 
		}
		void getCameraPose(std::vector<cv::Mat>& RList, std::vector<cv::Mat>& tList) { 
			RList = mRListMat; 
			tList = mtListMat; 
		}
		
		void getObjectPoints1(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<Eigen::Vector3d>& objectPoints);
		void getObjectPoints2(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<Eigen::Vector3d>& objectPoints);
		/*���㵥Ӧ����*/
		bool findHomography(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H, bool isNormal = false)throw(ZZYCalibrationFailure);
		bool findHomographyByRansac(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H, bool isNormal = false)throw(ZZYCalibrationFailure);
		bool findHomographyByOpenCV(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H)throw(ZZYCalibrationFailure);
	private:
		/*�궨ͼ������*/
		const int mImageNum;
		/*�궨��������·��*/
		std::string mStrCameraParaPath;
		/*�궨����2D&D������*/
		std::vector<std::vector<Eigen::Vector3d>> mObjectPoints;
		std::vector<std::vector<Eigen::Vector2d>> mImagePoints;
		/*�����������*/
		const int mDistortionParaNum;
		/*�ڲ�&�������&�������*/
		Eigen::Matrix3d mCameraMatrix;
		Eigen::Vector3d mRadialDistortion;
		Eigen::Vector2d mTangentialDistortion;
		/*�������*/
		std::vector<cv::Mat> mRListMat;
		std::vector<cv::Mat> mtListMat;
		/*�Ե�һ֡Ϊ�ο��õ�������������ϵ����*/
		std::vector<std::vector<Eigen::Vector3d>> mvX3D;
		/*��ӡ����ڲ�&���*/
		void PrintCameraIntrinsics();
	private:
		/*���ݹ�һ��*/
		bool Normalize(Eigen::MatrixXd&, Eigen::Matrix3d&);
		bool Normalize(const std::vector<Eigen::Vector2d>&, std::vector<Eigen::Vector2d>&, Eigen::Matrix3d&);
		/*VectorXd������PointXd�����໥ת��*/
		void Points2d2Vectors2d(const doublePoint2D&, doubleVector2D&);
		void Points3d2Vectors3d(const doublePoint3D&, doubleVector3D&);

		int rand_int(void) { return std::rand(); }
		Eigen::VectorXd constructVector(const Matrix3d& H, int i, int j);
		/*��ת��������ת����֮���໥ת��*/
		Eigen::Matrix3d RotationVector2Matrix(const Eigen::Vector3d&);
		Eigen::Vector3d RotationMatrix2Vector(const Eigen::Matrix3d&);
		/*��Ӧ�������*/
		Eigen::VectorXd solveHomographyDLT(const Eigen::MatrixXd& srcPoints, const Eigen::MatrixXd& dstPoints)throw(ZZYCalibrationFailure);
		/*����ʼ������ڲ�&���*/
		Eigen::Matrix3d solveInitCameraIntrinstic(const std::vector<Eigen::Matrix3d>&);
		void solveInitCameraExtrinstic(const std::vector<Matrix3d>&, const Eigen::Matrix3d&, std::vector<Eigen::Matrix3d>&, std::vector<Vector3d>&);
		
		void computeCameraCalibration(std::vector<std::vector<Eigen::Vector2d>>&,
			std::vector<std::vector<Eigen::Vector3d>>&,
			cv::Mat& cameraMatrix, cv::Mat&)throw(ZZYCalibrationFailure);
		double computeReprojectionErrors(const vector<vector<Eigen::Vector3d>>& objectPoints, const vector<vector<Eigen::Vector2d>>& imagePoints, const vector<Eigen::Vector3d>& rvecs, 
			const vector<Eigen::Vector3d>& tvecs, const Eigen::Matrix3d& cameraMatrix, const Eigen::VectorXd& distCoeffs, vector<double>& perViewErrors)throw(ZZYCalibrationFailure);
		/*��ά�ؽ���غ���*/
		double CalculateScale(const Eigen::Vector2d& imagePoint, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion,Eigen::Vector3d& x3D);
		void UndistortionKeys(const Eigen::Vector2d& vUnKeys, Eigen::Vector2d& vKeys, const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion);
		void Optimize(const Eigen::Matrix3d& K, std::vector<Eigen::Matrix3d>& RList, std::vector<Eigen::Vector3d>& tList, std::vector<std::vector<Eigen::Vector3d>>& vP3D);
		void Triangulate(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2, const Eigen::MatrixXd& P1, const Eigen::MatrixXd& P2, Eigen::Vector3d& x3D);
		void DepthRecover(const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion, const std::vector<Eigen::Matrix3d>& RList, const std::vector<Eigen::Vector3d>& tList, std::vector<std::vector<Eigen::Vector3d>>& vX3D);
	};
}