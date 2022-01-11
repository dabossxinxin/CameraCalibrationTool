#pragma once

#include <Eigen/Core>
#include <iostream>
#include <vector>

/*reference:Sun B,Zhu J,Yang L,et al.Calibration of line-scan 
cameras for precision measurement[J].Applied Optics,2016.*/
struct LineScanPara {
	float				vc;	/*线扫相机y向相机光心*/
	float				Fy;	/*线扫相机y向焦距*/
	Eigen::Matrix3f		R;	/*相机坐标系相对于世界坐标系的旋转*/
	Eigen::Vector3f		T;	/*相机坐标系相对于世界坐标系的平移*/
	float				k1;	/*线扫相机径向畸变参数*/
	float				k2;	/*线扫相机径向畸变参数*/
	float				k3;	/*线扫相机切向畸变参数*/
	float				resY;/*线扫相机分辨率*/
	bool				Conf;/*线扫相机分辨率置信度*/
	
	LineScanPara() {
		k1 = 0.;
		k2 = 0.;
		k3 = 0.;
	}
};

/*线扫相机重投影代价结构*/
struct LineScanProjectCost {
	Eigen::Vector3d objPt;
	Eigen::Vector2d imgPt;
	/*构造函数*/
	LineScanProjectCost(Eigen::Vector3d& objPt, Eigen::Vector2d& imgPt) :objPt(objPt), imgPt(imgPt) {}
	/*重载()操作符*/
	template <class T>
	bool operator()(
		const T* const k,
		const T* const r,
		const T* const t,
		T* residuals) const{
		T pos3d[3] = { T(objPt(0)),T(objPt(1)),T(objPt(2)) };
		T pos3d_proj[3] = { T(0.),T(0.),T(0.) };
		//旋转
		ceres::AngleAxisRotatePoint(r, pos3d, pos3d_proj);
		//平移
		pos3d_proj[0] += t[0];
		pos3d_proj[1] += t[1];
		pos3d_proj[2] += t[2];
		T xp = pos3d_proj[0] / pos3d_proj[2];
		T yp = pos3d_proj[1] / pos3d_proj[2];
		T xdis = T(0.0);
		T ydis = T(0.0);
		//线扫相机内参
		const T& fx = 1.;
		const T& cx = 0.;
		const T& fy = k[0];
		const T& cy = k[1];
		//线扫相机畸变参数
		const T& k1 = k[2];
		const T& k2 = k[3];
		const T& k3 = k[4];
		//添加径向畸变与切向畸变的影响
		xdis = xp;
		ydis = yp + k1*yp*yp*yp + k2*yp*yp*yp*yp*yp + k3*yp*yp;
		// 像素距离
		T u = fx*xdis + cx;
		T v = fy*ydis + cy;
		residuals[0] = u - T(imgPt(0));
		residuals[1] = v - T(imgPt(1));
		return true;
	}
};

class FeaturesPointExtract {
public:
	/*默认构造函数*/
	/*brief featuresNum:标定板上特征点的数量*/
	/*brief featuresHeight:标定板平面真实高度*/
	FeaturesPointExtract(const int featuresNum,const float featuresHeight) {
		mFeaturesNum = featuresNum;
		mFeatures2D.resize(mFeaturesNum,cv::Point2f(0.,0.));
		mFeatures3D.resize(mFeaturesNum,cv::Point3f(0.,0.,featuresHeight));
		mLineFunction2D.resize(mFeaturesNum,CommonStruct::LineFunction2D());
	};
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
	void SetCalibrationPara(const float& lineInterval, 
							const float& lineLength) 
	{
		mLineLength = lineLength;
		mLineInterval = lineInterval;
	}
	/*获取标定板的2D特征点*/
	void Get2DPoints(std::vector<cv::Point2f>& features2D) {
		features2D = mFeatures2D;
	}
	/*获取标定板的3D特征点*/
	void Get3DPoints(std::vector<cv::Point3f>& features3D) {
		features3D = mFeatures3D;
	}

	bool Update();

private:
	
	/*初始化特征提取程序*/
	/*1、3D特征点初始化*/
	/*2、标定板直线参数初始化*/
	/*3、提取标定板上的2D特征点*/
	void Initialize();

	/*计算指定几个点的交比*/
	/*brief 待计算交比的几个点*/
	bool CrossRatio(const std::vector<cv::Point2f>&,float);

	/*计算标定板上每条直线的直线方程*/
	bool BoardLineFunction();

	/*初始化标定板中vertical line的X坐标*/
	bool Features3DInitialize();

	/*提取图像特征点的像素坐标*/
	void CalculateFeatures2D();

	/*根据交比不变性计算斜线交点3D点坐标的X坐标部分*/
	float DiagonalLine3DPointX(const std::vector<cv::Point2f>&,
							  std::vector<cv::Point3f>&); 
	/*根据交比不变性计算斜线交点3D坐标*/
	void DiagonalLine3DPoints();
	
	/*根据斜线特征点拟合的直线方程求解直线特征点*/
	void VerticalLine3DPoints();
	
private:
	
	unsigned char*	mpFeatureImage;
	
	int				mImageWidth;
	
	int				mImageHeight;

	int				mFeaturesNum;

	float			mLineLength;
	
	float			mLineInterval;

	int				mCrossRatioPts = 4;

	std::vector<CommonStruct::LineFunction2D>		mLineFunction2D;

	std::vector<cv::Point2f>						mFeatures2D;
	
	std::vector<cv::Point3f>						mFeatures3D;
};

class LineScanCalibration {
public:
	/*默认构造函数*/
	LineScanCalibration() {};
	/*默认析构函数*/
	~LineScanCalibration() {};
	/*输入相机所有3D特征点坐标*/
	void SetObjectPoints(std::vector<std::vector<cv::Point3f>>& object) {
		mObjectPoints = object;
	}
	/*输入相机所有2D特征点坐标*/
	void SetImagePoints(std::vector<std::vector<cv::Point2f>>& points) {
		mImagePoints = points;
	}
	/*获取线扫相机标定参数*/
	void GetCameraPara(LineScanPara& para) {
		para = mCameraPara;
	}
	/*计算线扫相机标定参数*/
	bool Update();

private:
	/*通过拉格朗日条件极值获取初始相机参数*/
	bool InitialEstimate();
	/*通过非线性优化调优相机参数*/
	bool OptimizeEstimate();

	Eigen::MatrixXd skew(Eigen::MatrixXd& vec);

private:
	/*相机待标定参数*/
	LineScanPara							mCameraPara;
	
	/*世界坐标系特征点*/
	std::vector<std::vector<cv::Point3f>>	mObjectPoints;
	/*图像像素坐标系特征点*/
	std::vector<std::vector<cv::Point2f>>	mImagePoints;
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
};