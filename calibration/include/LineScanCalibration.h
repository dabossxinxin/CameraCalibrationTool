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

class LineScanCalibration {
public:
	/*默认构造函数*/
	LineScanCalibration() {};
	/*默认析构函数*/
	~LineScanCalibration() {};
	/*输入相机所有3D特征点坐标*/
	void SetObjectPoints(std::vector<std::vector<Eigen::Vector3f>>& object) {
		mObjectPoints = object;
	}
	/*输入相机所有2D特征点坐标*/
	void SetImagePoints(std::vector<std::vector<Eigen::Vector2f>>& points) {
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

private:
	/*相机待标定参数*/
	LineScanPara								mCameraPara;
	
	/*世界坐标系特征点*/
	std::vector<std::vector<Eigen::Vector3f>>	mObjectPoints;
	/*图像像素坐标系特征点*/
	std::vector<std::vector<Eigen::Vector2f>>	mImagePoints;

};