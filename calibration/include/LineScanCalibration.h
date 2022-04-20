#pragma once

#include <Eigen/Core>
#include <iostream>
#include <vector>

#include "CommonFunctions.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <opencv2/opencv.hpp>

/*reference:Sun B,Zhu J,Yang L,et al.Calibration of line-scan 
cameras for precision measurement[J].Applied Optics,2016.*/
struct LineScanPara {
	double								vc;		/*��ɨ���y���������*/
	double								Fy;		/*��ɨ���y�򽹾�*/
	Eigen::Matrix3d						R;		/*�������ϵ�������������ϵ����ת*/
	Eigen::Vector3d						T;		/*�������ϵ�������������ϵ��ƽ��*/
	double								k1;		/*��ɨ�������������*/
	double								k2;		/*��ɨ�������������*/
	double								k3;		/*��ɨ�������������*/
	double								resY;	/*��ɨ����ֱ���*/                    
	bool								Conf;	/*��ɨ����ֱ������Ŷ�*/
	
	LineScanPara() {
		k1 = 0.;
		k2 = 0.;
		k3 = 0.;
		vc = 0.;
		Fy = 0.;
	}
};

struct InstrinsticPara {
	double vc;
	double Fy;
};

struct ExtrinsticPara {
	Eigen::Matrix3d	R;
	Eigen::Vector3d	T;
};

/*��ɨ�����ͶӰ���۽ṹ*/
struct LineScanProjectCost {
	cv::Point3d objPt;
	cv::Point2d imgPt;
	/*���캯��*/
	LineScanProjectCost(cv::Point3d& objPt, cv::Point2d& imgPt) :objPt(objPt), imgPt(imgPt) {}
	/*����()������*/
	template <class T>
	bool operator()(
		const T* const k,
		const T* const r,
		const T* const t,
		T* residuals) const {
		T pos3d[3] = { T(objPt.x),T(objPt.y),T(objPt.z) };
		T pos3d_proj[3];
		//��ת
		ceres::AngleAxisRotatePoint(r, pos3d, pos3d_proj);
		//ƽ��
		pos3d_proj[0] += t[0];
		pos3d_proj[1] += t[1];
		pos3d_proj[2] += t[2];
		//��һ���������ϵ
		T xp = pos3d_proj[0] / pos3d_proj[2];
		T yp = pos3d_proj[1] / pos3d_proj[2];
		//��ɨ����ڲ�
		const T& fx = T(1.);
		const T& cx = T(0.);
		const T& fy = k[0];
		const T& cy = k[1];
		//��ɨ����������
		const T& k1 = k[2];
		const T& k2 = k[3];
		const T& k3 = k[4];
		//��Ӿ����������������Ӱ��
		T xdis = xp;
		T ydis = yp + k1*yp*yp*yp + k2*yp*yp*yp*yp*yp /*+ k3*yp*yp*/;
		// ���ؾ���
		T u = fx*xdis + cx;
		T v = fy*ydis + cy;
		residuals[0] = /*(ceres::abs)*/(u - T(imgPt.x));
		residuals[1] = /*(ceres::abs)*/(v - T(imgPt.y));
		return true;
	}
};

class FeaturesPointExtract {
public:
	/*Ĭ�Ϲ��캯��*/
	/*brief featuresNum:�궨���������������*/
	/*brief featuresHeight:�궨��ƽ����ʵ�߶�*/
	FeaturesPointExtract(const int featuresNum,const double featuresHeight) {
		mFeaturesNum = featuresNum;
		mFeatures2D.resize(mFeaturesNum,cv::Point2d(0.,0.));
		mFeatures3D.resize(mFeaturesNum,cv::Point3d(0.,0.,featuresHeight));
		mLineFunction2D.resize(mFeaturesNum,CommonStruct::LineFunction2D());
	};
	/*ͼ��������ȥ�����ʹ�øù��캯���Ż�3D��*/
	FeaturesPointExtract(const std::vector<cv::Point2d>& features2D, const int featuresNum, const double featuresHeight) {
		mFeaturesNum = featuresNum;
		mFeatures2D = features2D;
		mFeatures3D.resize(mFeaturesNum, cv::Point3d(0., 0., featuresHeight));
		mLineFunction2D.resize(mFeaturesNum, CommonStruct::LineFunction2D());
	}
	/*Ĭ����������*/
	~FeaturesPointExtract() {};
	/*����ͼ��*/
	/*Warning:ͼ��ָ�����ⲿ���������ⲿ�ͷ�*/
	void SetFeatureImage(unsigned char* const pImage, 
						 const int width,
						 const int height) 
	{
		mImageHeight = height;
		mImageWidth = width;
		mpFeatureImage = pImage;
	}
	/*���ñ궨��Ĳ���*/
	void SetCalibrationPara(const double& lineInterval, 
							const double& lineLength) 
	{
		mLineLength = lineLength;
		mLineInterval = lineInterval;
	}
	/*���õ�����Ϣ�����·��*/
	void SetDebugPath(const std::string& path) {
		mDebugPath = path;
	}
	/*��ȡ�궨���2D������*/
	void Get2DPoints(std::vector<cv::Point2d>& features2D) {
		features2D = mFeatures2D;
	}
	/*��ȡ�궨���3D������*/
	void Get3DPoints(std::vector<cv::Point3d>& features3D) {
		features3D = mFeatures3D;
	}
	/*2D������δ֪ʱ���øú���*/
	bool Update();
	/*��֪2D������ʱ���øú���*/
	bool UpdateWithFeatures();

private:
	
	/*��ʼ��������ȡ����*/
	/*1��3D�������ʼ��*/
	/*2���궨��ֱ�߲�����ʼ��*/
	/*3����ȡ�궨���ϵ�2D������*/
	void Initialize();
	/*��ʼ��������ȡ����*/
	/*1��3D�������ʼ��*/
	/*2���궨��ֱ�߲�����ʼ��*/
	void InitializeWithFeatures();
	/*����ָ��������Ľ���*/
	/*brief �����㽻�ȵļ�����*/
	bool CrossRatio(const std::vector<cv::Point2d>&,double&);
	bool CrossRatio_L(const std::vector<cv::Point2d>&,double&);
	/*����궨����ÿ��ֱ�ߵ�ֱ�߷���*/
	bool BoardLineFunction();
	/*��ʼ���궨����vertical line��X����*/
	bool Features3DInitialize();
	/*��ȡͼ�����������������*/
	void CalculateFeatures2D();
	/*���ݽ��Ȳ����Լ���б�߽���3D�������X���겿��*/
	double DiagonalLine3DPointX(const std::vector<cv::Point2d>&,
							  std::vector<cv::Point3d>&); 
	double DiagonalLine3DPointX_L(const std::vector<cv::Point2d>&,
							std::vector<cv::Point3d>&);
	/*���ݽ��Ȳ����Լ���б�߽���3D����*/
	void DiagonalLine3DPoints();
	/*����б����������ϵ�ֱ�߷������ֱ��������*/
	void VerticalLine3DPoints();
	/*���ݲ�������Debugͼ��*/
	void GenerateDebugImage();
	
private:
	std::string		mDebugPath;
	
	unsigned char*	mpFeatureImage;
	
	int				mImageWidth;
	
	int				mImageHeight;

	int				mFeaturesNum;

	double			mLineLength;
	
	double			mLineInterval;

	int				mCrossRatioPts = 4;

	std::vector<CommonStruct::LineFunction2D>		mLineFunction2D;

	std::vector<cv::Point2d>						mFeatures2D;
	
	std::vector<cv::Point3d>						mFeatures3D;

	cv::Mat											mImageDebug;
};

class LineScanCalibration {
public:
	/*Ĭ�Ϲ��캯��*/
	LineScanCalibration() {
		mCameraPara.Conf = true;
	};
	/*Ĭ����������*/
	~LineScanCalibration() {};
	/*�����������3D����������*/
	void SetObjectPoints(std::vector<std::vector<cv::Point3d>>& object) {
		mObjectPoints = object;
	}
	/*�����������2D����������*/
	void SetImagePoints(std::vector<std::vector<cv::Point2d>>& points) {
		mImagePoints = points;
	}
	/*���õ�����Ϣ�����·��*/
	void SetDebugPath(const std::string& path) {
		mDebugPath = path;
	}
	/*��ȡ��ɨ����궨����*/
	void GetCameraPara(LineScanPara& para) {
		para = mCameraPara;
	}
	void GetInitCameraPara(std::vector<LineScanPara>& para) {
		para = mIniCameraPara;
	}
	/*��ȡ�������ϵ3D��*/
	void GetWorldPointsBeforeOptimized(std::vector<cv::Point3d>& pts) {
		pts = mWorldPointsBeforeOptimized;
	}
	void GetWorldPointsAfterOptimized(std::vector<cv::Point3d>& pts) {
		pts = mWorldPointsAfterOptimized;
	}
	void GetGroundTruth(std::vector<cv::Point3d>& pts) {
		pts = mGroudTruth;
	}
	/*������ɨ����궨����*/
	bool Update();

private:
	/*ͨ����������������ֵ��ȡ��ʼ�������*/
	bool InitialEstimate();
	bool InitialEstimate2();
	bool InitialEstimate3();
	/*ͨ���������Ż������������*/
	bool OptimizeEstimate();
	/*�õ����Գƾ���*/
	Eigen::MatrixXd skew(Eigen::MatrixXd& vec);
	Eigen::Matrix3f skew(Eigen::Matrix<float, 1, 3>& vec);
	Eigen::Matrix3d skew(Eigen::Matrix<double, 1, 3>& vec);
	/*��M����ת��Ϊ����ڲ��Լ����*/
	void ReshapePara(const Eigen::Matrix<double, 2, 4>& M, const Eigen::Vector3d& sumPts, InstrinsticPara& paraIn, ExtrinsticPara& paraEx);
	/*������ͶӰ���*/
	double ReprojectError(std::vector<cv::Point2d>&, std::vector<cv::Point3d>&, Eigen::Matrix3d&, Eigen::Vector3d&,Eigen::Vector3d&,double&, double&);
	double ReprojectError(std::vector<cv::Point2d>& pt2D, std::vector<cv::Point3d>& pt3D, Eigen::Matrix<double, 2, 4>& M);
	/*���㵥Ŀ��ɨ����ĳ߶�*/
	double CalculateScale(cv::Point2d&, Eigen::Matrix3d&, Eigen::Vector3d&, Eigen::Matrix3d&, Eigen::Vector3d&, cv::Point3d&, double);
	/*����������Ļ���*/
	void LineScanCalibration::UndistortionKeys(Eigen::Vector2d&, Eigen::Vector2d&, Eigen::Matrix3d&, Eigen::Vector3d&);
	/*��������ķֱ��������Ŷ�*/
	bool Resolution();
	/*����ת����ת��Ϊ��ת����*/
	Eigen::Vector3d RotationMatrix2Vector(const Eigen::Matrix3d&);
	/*����ת����ת��Ϊ��ת����*/
	Eigen::Matrix3d RotationVector2Matrix(const Eigen::Vector3d&);
	/*��double���͵ľ���ת��Ϊfloat���͵ľ���*/
	Eigen::Matrix3f Matrixd2f(const Eigen::Matrix3d&);
	/*����debugͼ��*/
	void SaveDebugImage();
	/*�������������*/
	void SaveGroundTruth();
	void SaveWorldPointsBeforeOptimized(LineScanPara& cameraPara);
	void SaveWorldPointsAfterOptimized(LineScanPara& cameraPara);
private:
	/*����ؼ���Ϣ��·��*/
	std::string								mDebugPath;
	/*������궨����*/
	LineScanPara							mCameraPara;
	std::vector<LineScanPara>				mIniCameraPara;
	/*��������ϵ������*/
	std::vector<std::vector<cv::Point3d>>	mObjectPoints;
	/*ȥ����������������ϵ������*/
	std::vector<std::vector<cv::Point3d>>	mObjectPointsDeDis;
	/*ͼ����������ϵ������*/
	std::vector<std::vector<cv::Point2d>>	mImagePoints;
	/*ȥ��������ͼ����������ϵ������*/
	std::vector<std::vector<cv::Point2d>>	mImagePointsDeDis;    
	/*��������궨���������Ӧ�����������*/
	std::vector<cv::Point3d>				mWorldPointsBeforeOptimized;
	std::vector<cv::Point3d>				mWorldPointsAfterOptimized;
	std::vector<cv::Point3d>				mGroudTruth;
	/*������*/
	cv::Mat									mImageDebug;
	/*ȥ����ͼ��*/
	cv::Mat									mUndistortImage;
};