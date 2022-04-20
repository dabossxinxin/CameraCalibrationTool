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
	double								vc;		/*线扫相机y向相机光心*/
	double								Fy;		/*线扫相机y向焦距*/
	Eigen::Matrix3d						R;		/*相机坐标系相对于世界坐标系的旋转*/
	Eigen::Vector3d						T;		/*相机坐标系相对于世界坐标系的平移*/
	double								k1;		/*线扫相机径向畸变参数*/
	double								k2;		/*线扫相机径向畸变参数*/
	double								k3;		/*线扫相机切向畸变参数*/
	double								resY;	/*线扫相机分辨率*/                    
	bool								Conf;	/*线扫相机分辨率置信度*/
	
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

/*线扫相机重投影代价结构*/
struct LineScanProjectCost {
	cv::Point3d objPt;
	cv::Point2d imgPt;
	/*构造函数*/
	LineScanProjectCost(cv::Point3d& objPt, cv::Point2d& imgPt) :objPt(objPt), imgPt(imgPt) {}
	/*重载()操作符*/
	template <class T>
	bool operator()(
		const T* const k,
		const T* const r,
		const T* const t,
		T* residuals) const {
		T pos3d[3] = { T(objPt.x),T(objPt.y),T(objPt.z) };
		T pos3d_proj[3];
		//旋转
		ceres::AngleAxisRotatePoint(r, pos3d, pos3d_proj);
		//平移
		pos3d_proj[0] += t[0];
		pos3d_proj[1] += t[1];
		pos3d_proj[2] += t[2];
		//归一化相机坐标系
		T xp = pos3d_proj[0] / pos3d_proj[2];
		T yp = pos3d_proj[1] / pos3d_proj[2];
		//线扫相机内参
		const T& fx = T(1.);
		const T& cx = T(0.);
		const T& fy = k[0];
		const T& cy = k[1];
		//线扫相机畸变参数
		const T& k1 = k[2];
		const T& k2 = k[3];
		const T& k3 = k[4];
		//添加径向畸变与切向畸变的影响
		T xdis = xp;
		T ydis = yp + k1*yp*yp*yp + k2*yp*yp*yp*yp*yp /*+ k3*yp*yp*/;
		// 像素距离
		T u = fx*xdis + cx;
		T v = fy*ydis + cy;
		residuals[0] = /*(ceres::abs)*/(u - T(imgPt.x));
		residuals[1] = /*(ceres::abs)*/(v - T(imgPt.y));
		return true;
	}
};

class FeaturesPointExtract {
public:
	/*默认构造函数*/
	/*brief featuresNum:标定板上特征点的数量*/
	/*brief featuresHeight:标定板平面真实高度*/
	FeaturesPointExtract(const int featuresNum,const double featuresHeight) {
		mFeaturesNum = featuresNum;
		mFeatures2D.resize(mFeaturesNum,cv::Point2d(0.,0.));
		mFeatures3D.resize(mFeaturesNum,cv::Point3d(0.,0.,featuresHeight));
		mLineFunction2D.resize(mFeaturesNum,CommonStruct::LineFunction2D());
	};
	/*图像特征点去畸变后使用该构造函数优化3D点*/
	FeaturesPointExtract(const std::vector<cv::Point2d>& features2D, const int featuresNum, const double featuresHeight) {
		mFeaturesNum = featuresNum;
		mFeatures2D = features2D;
		mFeatures3D.resize(mFeaturesNum, cv::Point3d(0., 0., featuresHeight));
		mLineFunction2D.resize(mFeaturesNum, CommonStruct::LineFunction2D());
	}
	/*默认析构函数*/
	~FeaturesPointExtract() {};
	/*输入图像*/
	/*Warning:图像指针在外部创建，在外部释放*/
	void SetFeatureImage(unsigned char* const pImage, 
						 const int width,
						 const int height) 
	{
		mImageHeight = height;
		mImageWidth = width;
		mpFeatureImage = pImage;
	}
	/*设置标定板的参数*/
	void SetCalibrationPara(const double& lineInterval, 
							const double& lineLength) 
	{
		mLineLength = lineLength;
		mLineInterval = lineInterval;
	}
	/*设置调试信息输出的路径*/
	void SetDebugPath(const std::string& path) {
		mDebugPath = path;
	}
	/*获取标定板的2D特征点*/
	void Get2DPoints(std::vector<cv::Point2d>& features2D) {
		features2D = mFeatures2D;
	}
	/*获取标定板的3D特征点*/
	void Get3DPoints(std::vector<cv::Point3d>& features3D) {
		features3D = mFeatures3D;
	}
	/*2D特征点未知时调用该函数*/
	bool Update();
	/*已知2D特征点时调用该函数*/
	bool UpdateWithFeatures();

private:
	
	/*初始化特征提取程序*/
	/*1、3D特征点初始化*/
	/*2、标定板直线参数初始化*/
	/*3、提取标定板上的2D特征点*/
	void Initialize();
	/*初始化特征提取程序*/
	/*1、3D特征点初始化*/
	/*2、标定板直线参数初始化*/
	void InitializeWithFeatures();
	/*计算指定几个点的交比*/
	/*brief 待计算交比的几个点*/
	bool CrossRatio(const std::vector<cv::Point2d>&,double&);
	bool CrossRatio_L(const std::vector<cv::Point2d>&,double&);
	/*计算标定板上每条直线的直线方程*/
	bool BoardLineFunction();
	/*初始化标定板中vertical line的X坐标*/
	bool Features3DInitialize();
	/*提取图像特征点的像素坐标*/
	void CalculateFeatures2D();
	/*根据交比不变性计算斜线交点3D点坐标的X坐标部分*/
	double DiagonalLine3DPointX(const std::vector<cv::Point2d>&,
							  std::vector<cv::Point3d>&); 
	double DiagonalLine3DPointX_L(const std::vector<cv::Point2d>&,
							std::vector<cv::Point3d>&);
	/*根据交比不变性计算斜线交点3D坐标*/
	void DiagonalLine3DPoints();
	/*根据斜线特征点拟合的直线方程求解直线特征点*/
	void VerticalLine3DPoints();
	/*根据参数生成Debug图像*/
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
	/*默认构造函数*/
	LineScanCalibration() {
		mCameraPara.Conf = true;
	};
	/*默认析构函数*/
	~LineScanCalibration() {};
	/*输入相机所有3D特征点坐标*/
	void SetObjectPoints(std::vector<std::vector<cv::Point3d>>& object) {
		mObjectPoints = object;
	}
	/*输入相机所有2D特征点坐标*/
	void SetImagePoints(std::vector<std::vector<cv::Point2d>>& points) {
		mImagePoints = points;
	}
	/*设置调试信息输出的路径*/
	void SetDebugPath(const std::string& path) {
		mDebugPath = path;
	}
	/*获取线扫相机标定参数*/
	void GetCameraPara(LineScanPara& para) {
		para = mCameraPara;
	}
	void GetInitCameraPara(std::vector<LineScanPara>& para) {
		para = mIniCameraPara;
	}
	/*获取相机坐标系3D点*/
	void GetWorldPointsBeforeOptimized(std::vector<cv::Point3d>& pts) {
		pts = mWorldPointsBeforeOptimized;
	}
	void GetWorldPointsAfterOptimized(std::vector<cv::Point3d>& pts) {
		pts = mWorldPointsAfterOptimized;
	}
	void GetGroundTruth(std::vector<cv::Point3d>& pts) {
		pts = mGroudTruth;
	}
	/*计算线扫相机标定参数*/
	bool Update();

private:
	/*通过拉格朗日条件极值获取初始相机参数*/
	bool InitialEstimate();
	bool InitialEstimate2();
	bool InitialEstimate3();
	/*通过非线性优化调优相机参数*/
	bool OptimizeEstimate();
	/*得到反对称矩阵*/
	Eigen::MatrixXd skew(Eigen::MatrixXd& vec);
	Eigen::Matrix3f skew(Eigen::Matrix<float, 1, 3>& vec);
	Eigen::Matrix3d skew(Eigen::Matrix<double, 1, 3>& vec);
	/*将M矩阵转化为相机内参以及外参*/
	void ReshapePara(const Eigen::Matrix<double, 2, 4>& M, const Eigen::Vector3d& sumPts, InstrinsticPara& paraIn, ExtrinsticPara& paraEx);
	/*计算重投影误差*/
	double ReprojectError(std::vector<cv::Point2d>&, std::vector<cv::Point3d>&, Eigen::Matrix3d&, Eigen::Vector3d&,Eigen::Vector3d&,double&, double&);
	double ReprojectError(std::vector<cv::Point2d>& pt2D, std::vector<cv::Point3d>& pt3D, Eigen::Matrix<double, 2, 4>& M);
	/*计算单目线扫相机的尺度*/
	double CalculateScale(cv::Point2d&, Eigen::Matrix3d&, Eigen::Vector3d&, Eigen::Matrix3d&, Eigen::Vector3d&, cv::Point3d&, double);
	/*消除特征点的畸变*/
	void LineScanCalibration::UndistortionKeys(Eigen::Vector2d&, Eigen::Vector2d&, Eigen::Matrix3d&, Eigen::Vector3d&);
	/*计算相机的分辨率与置信度*/
	bool Resolution();
	/*将旋转矩阵转化为旋转向量*/
	Eigen::Vector3d RotationMatrix2Vector(const Eigen::Matrix3d&);
	/*将旋转向量转化为旋转矩阵*/
	Eigen::Matrix3d RotationVector2Matrix(const Eigen::Vector3d&);
	/*将double类型的矩阵转化为float类型的矩阵*/
	Eigen::Matrix3f Matrixd2f(const Eigen::Matrix3d&);
	/*保存debug图像*/
	void SaveDebugImage();
	/*计算世界坐标点*/
	void SaveGroundTruth();
	void SaveWorldPointsBeforeOptimized(LineScanPara& cameraPara);
	void SaveWorldPointsAfterOptimized(LineScanPara& cameraPara);
private:
	/*保存关键信息的路径*/
	std::string								mDebugPath;
	/*相机待标定参数*/
	LineScanPara							mCameraPara;
	std::vector<LineScanPara>				mIniCameraPara;
	/*世界坐标系特征点*/
	std::vector<std::vector<cv::Point3d>>	mObjectPoints;
	/*去除畸变后的世界坐标系特征点*/
	std::vector<std::vector<cv::Point3d>>	mObjectPointsDeDis;
	/*图像像素坐标系特征点*/
	std::vector<std::vector<cv::Point2d>>	mImagePoints;
	/*去除畸变后的图像像素坐标系特征点*/
	std::vector<std::vector<cv::Point2d>>	mImagePointsDeDis;    
	/*参与相机标定的特征点对应的世界坐标点*/
	std::vector<cv::Point3d>				mWorldPointsBeforeOptimized;
	std::vector<cv::Point3d>				mWorldPointsAfterOptimized;
	std::vector<cv::Point3d>				mGroudTruth;
	/*调试用*/
	cv::Mat									mImageDebug;
	/*去畸变图像*/
	cv::Mat									mUndistortImage;
};