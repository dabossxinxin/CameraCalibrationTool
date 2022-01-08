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

class FeaturesPointExtract {
public:
	/*Ĭ�Ϲ��캯��*/
	/*brief featuresNum:�궨���������������*/
	/*brief featuresHeight:�궨��ƽ����ʵ�߶�*/
	FeaturesPointExtract(const int featuresNum,const float featuresHeight) {
		mFeaturesNum = featuresNum;
		mFeatures2D.resize(mFeaturesNum,cv::Point2f(0.,0.));
		mFeatures3D.resize(mFeaturesNum,cv::Point3f(0.,0.,featuresHeight));
		mLineFunction2D.resize(mFeaturesNum,CommonStruct::LineFunction2D());
	};
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
	void SetCalibrationPara(const float& lineInterval, 
							const float& lineLength) 
	{
		mLineLength = lineLength;
		mLineInterval = lineInterval;
	}
	/*��ȡ�궨���2D������*/
	void Get2DPoints(std::vector<cv::Point2f>& features2D) {
		features2D = mFeatures2D;
	}
	/*��ȡ�궨���3D������*/
	void Get3DPoints(std::vector<cv::Point3f>& features3D) {
		features3D = mFeatures3D;
	}

	bool Update();

private:
	
	/*��ʼ��������ȡ����*/
	/*1��3D�������ʼ��*/
	/*2���궨��ֱ�߲�����ʼ��*/
	/*3����ȡ�궨���ϵ�2D������*/
	void Initialize();

	/*����ָ��������Ľ���*/
	/*brief �����㽻�ȵļ�����*/
	bool CrossRatio(const std::vector<cv::Point2f>&,float);

	/*����궨����ÿ��ֱ�ߵ�ֱ�߷���*/
	/*brief featuresNum:�궨�����߶ε�����*/
	/*brief lineInterval:�궨�����߶εļ��*/
	/*brief lineLength:�궨�����߶εĳ���*/
	bool BoardLineFunction(const int featuresNum, 
						   const float lineInterval,
						   const float lineLength);

	/*��ʼ���궨����vertical line��X����*/
	/*brief featuresNum:�궨�����߶ε�����*/
	/*brief lineInterval:�궨�����߶εļ��*/
	/*brief lineLength:�궨�����߶εĳ���*/
	bool Features3DInitialize(const int featuresNum,
							  const float lineInterval,
							  const float lineLength);

	/*��ȡͼ�����������������*/
	/*brief pImage:����ȡ������ͼ��*/
	/*brief width:����ȡ����ͼ������ؿ��*/
	/*brief height:����ȡ����ͼ������ظ߶�*/
	void CalculateFeatures2D(unsigned char* const pImage,
							 const int width,
							 const int height);

	/*���ݽ��Ȳ����Լ���б�߽���3D�������X���겿��*/
	float DiagonalLine3DPointX(const std::vector<cv::Point2f>&,
							  std::vector<cv::Point3f>&); 
	/*���ݽ��Ȳ����Լ���б�߽���3D����*/
	void DiagonalLine3DPoints(const std::vector<cv::Point2f>&,
							  std::vector<cv::Point3f>&);
	
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