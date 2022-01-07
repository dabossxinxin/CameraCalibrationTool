#pragma once

#include <Eigen/Core>
#include <iostream>
#include <vector>

/*reference:Sun B,Zhu J,Yang L,et al.Calibration of line-scan 
cameras for precision measurement[J].Applied Optics,2016.*/
struct LineScanPara {
	float				vc;	/*��ɨ���y���������*/
	float				Fy;	/*��ɨ���y�򽹾�*/
	Eigen::Matrix3f		R;	/*�������ϵ�������������ϵ����ת*/
	Eigen::Vector3f		T;	/*�������ϵ�������������ϵ��ƽ��*/
	float				k1;	/*��ɨ�������������*/
	float				k2;	/*��ɨ�������������*/
	float				k3;	/*��ɨ�������������*/
	float				resY;/*��ɨ����ֱ���*/
	bool				Conf;/*��ɨ����ֱ������Ŷ�*/
	
	LineScanPara() {
		k1 = 0.;
		k2 = 0.;
		k3 = 0.;
	}
};

class LineScanCalibration {
public:
	/*Ĭ�Ϲ��캯��*/
	LineScanCalibration() {};
	/*Ĭ����������*/
	~LineScanCalibration() {};
	/*�����������3D����������*/
	void SetObjectPoints(std::vector<std::vector<Eigen::Vector3f>>& object) {
		mObjectPoints = object;
	}
	/*�����������2D����������*/
	void SetImagePoints(std::vector<std::vector<Eigen::Vector2f>>& points) {
		mImagePoints = points;
	}
	/*��ȡ��ɨ����궨����*/
	void GetCameraPara(LineScanPara& para) {
		para = mCameraPara;
	}
	/*������ɨ����궨����*/
	bool Update();

private:
	/*ͨ����������������ֵ��ȡ��ʼ�������*/
	bool InitialEstimate();
	/*ͨ���������Ż������������*/
	bool OptimizeEstimate();

private:
	/*������궨����*/
	LineScanPara								mCameraPara;
	
	/*��������ϵ������*/
	std::vector<std::vector<Eigen::Vector3f>>	mObjectPoints;
	/*ͼ����������ϵ������*/
	std::vector<std::vector<Eigen::Vector2f>>	mImagePoints;

};