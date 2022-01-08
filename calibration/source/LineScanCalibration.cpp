#include "CommonFunctions.h"
#include "LineScanCalibration.h"

bool FeaturesPointExtract::CrossRatio(const std::vector<cv::Point2f>& features,float crossRatio) 
{
	if (features.size() != mCrossRatioPts) return false;
	const float v1 = features[0].y;
	const float v2 = features[1].y;
	const float v3 = features[2].y;
	const float v4 = features[3].y;
	crossRatio = ((v1-v3)*(v2-v4))/((v2-v3)*(v1-v4));
}

float FeaturesPointExtract::DiagonalLine3DPoint(const std::vector<cv::Point2f>& features2D,
											    std::vector<cv::Point3f>& features3D)
{
	if (features2D.size() != mCrossRatioPts) return;
	if (features3D.size() != mCrossRatioPts) return;

	const float x1 = features3D[0].x;
	const float x2 = features3D[1].x;
	const float x3 = features3D[2].x;
	const float x4 = features3D[3].x;
	
	float crossRatio = 0.;
	this->CrossRatio(features2D, crossRatio);
	float rho = crossRatio*(x1-x4)/(x1-x3);
	float xCor = (x4-rho*x3)/(1-rho);
}

bool FeaturesPointExtract::BoardLineFunction(const int featuresNum,
											 const float lineInterval,
											 const float lineLength)
{
	const int Num = mLineFunction2D.size();
	if (Num != featuresNum) return false;
	const float halfInterval = 0.5*lineInterval;

	for (size_t it = 0; it < Num; ++it) {
		//vertical line
		if (it % 2 == 0) {
			mLineFunction2D[it].a = 1.0;
			mLineFunction2D[it].b = 0.;
			mLineFunction2D[it].c = -halfInterval*it;
		}
		//diagonal line
		else {
			cv::Point2f p1((it-1)*halfInterval,0.);
			cv::Point2f p2((it+1)*halfInterval,lineLength);
			mLineFunction2D[it] = CommonFunctions::ComputeLineFunction2D(p1, p2);
		}
	}
	return true;
}

void FeaturesPointExtract::CalculateFeatures2D(unsigned char* const pImage,
											   const int width,
											   const int height)
{
	return;
}

bool FeaturesPointExtract::Update()
{ 
	//��ȡ�궨��ͼ���ϵ�2D������
	this->CalculateFeatures2D(mpFeatureImage, mImageWidth, mImageHeight);

	//����궨��������ֱ�ߵ�ֱ�߷���
	this->BoardLineFunction(mFeaturesNum, mLineInterval, mLineLength);
	
	//ͨ�����Ȳ����Լ���diagonal line 3D�������X����
	
	//ͨ��ֱ�߷��̼���diagonal line 3D�������Y����

	//ʹ��diagonal line 3D������X��Y�������ֱ�߷���

	//ͨ����ϵ�ֱ�߷��̼���vertical line 3D�������Y����
	return true;
}

bool LineScanCalibration::Update() 
{
	// �жϲ����Ƿ���ȷ����
	if (mObjectPoints.empty() || mImagePoints.empty()) {
		std::cerr << "Object Points || Image Points Size Empty..." << std::endl;
		return false;
	}
	if (mObjectPoints.size() != mImagePoints.size()) {
		std::cerr << "Object Points's size not equal to Image Points's..." << std::endl;
		return false;
	}

	if (!this->InitialEstimate()) {
		std::cerr << "Initial Estimate Failed..." << std::endl;
		return false;
	}
	if (!this->OptimizeEstimate()) {
		std::cerr << "Optimize Estimate Failed..." << std::endl;
		return false;
	}

	return true;
}

bool LineScanCalibration::InitialEstimate() 
{
	CommonFunctions::ConditionPrint("Start Initial Estimate...");

	CommonFunctions::ConditionPrint("End Initial Estimate");
	return true;
}

bool LineScanCalibration::OptimizeEstimate() 
{
	CommonFunctions::ConditionPrint("Start Optimize Estimate...");

	CommonFunctions::ConditionPrint("End Optimize Estimate");
	return true;
}