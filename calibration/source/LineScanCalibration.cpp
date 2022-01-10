#include "CommonFunctions.h"
#include "LineScanCalibration.h"
#include "GrowAllSpecifiedRegions.h"

void FeaturesPointExtract::Initialize()
{
	//获取函数运行必要的参数
	const int featuresNum = this->mFeaturesNum;
	const float lineInterval = this->mLineInterval;
	const float lineLength = this->mLineLength;
	const int imageWidth = this->mImageWidth;
	const int imageHeight = this->mImageHeight;
	const int imageSize = imageWidth*imageHeight;

	//获取标定板上的2D特征点坐标
	this->CalculateFeatures2D();
	//初始化3D点 vertical line上的X坐标
	this->Features3DInitialize();
	//初始化标定板上所有直线的直线方程
	this->BoardLineFunction();
}

bool FeaturesPointExtract::CrossRatio(const std::vector<cv::Point2f>& features,float crossRatio) 
{
	if (features.size() != mCrossRatioPts) return false;
	const float v1 = features[0].y;
	const float v2 = features[1].y;
	const float v3 = features[2].y;
	const float v4 = features[3].y;
	crossRatio = ((v1-v3)*(v2-v4))/((v2-v3)*(v1-v4));
}

float FeaturesPointExtract::DiagonalLine3DPointX(const std::vector<cv::Point2f>& features2D,
											    std::vector<cv::Point3f>& features3D)
{
	if (features2D.size() != mCrossRatioPts) exit(-1);
	if (features3D.size() != mCrossRatioPts) exit(-1);

	const float x1 = features3D[0].x;
	const float x2 = features3D[1].x;
	const float x3 = features3D[2].x;
	const float x4 = features3D[3].x;
	
	float crossRatio = 0.;
	this->CrossRatio(features2D, crossRatio);
	float rho = crossRatio*(x1-x4)/(x1-x3);
	return (x4-rho*x3)/(1-rho);
}

void FeaturesPointExtract::DiagonalLine3DPoints(const std::vector<cv::Point2f>& imagePos,
												std::vector<cv::Point3f>& worldPos)
{
	if (imagePos.size() != mFeaturesNum) exit(-1);
	if (worldPos.size() != mFeaturesNum) exit(-1);

	for (size_t it = 0; it < mFeaturesNum; ++it) {
		//diagonal line
		if (it % 2 != 0) {
			std::vector<cv::Point2f> tmp2D; tmp2D.clear();
			std::vector<cv::Point3f> tmp3D; tmp3D.clear();
			bool flag2D = false;
			if (it - 3 >= 0) {
				flag2D = true;
				tmp2D.push_back(mFeatures2D[it - 3]);
			}
			tmp2D.push_back(mFeatures2D[it - 1]);
			tmp2D.push_back(mFeatures2D[it]);
			tmp2D.push_back(mFeatures2D[it + 1]);
			if (!flag2D) tmp2D.push_back(mFeatures2D[it + 3]);
			if (tmp2D.size() != mCrossRatioPts) exit(-1);
			bool flag3D = false;
			if (it - 3 >= 0) {
				flag3D = true;
				tmp3D.push_back(mFeatures3D[it - 3]);
			}

			tmp3D.push_back(mFeatures3D[it - 1]);
			tmp3D.push_back(mFeatures3D[it]);
			tmp3D.push_back(mFeatures3D[it + 1]);
			if (!flag3D) tmp3D.push_back(mFeatures3D[it + 3]);	
			if (tmp3D.size() != mCrossRatioPts) exit(-1);
			mFeatures3D[it].x = this->DiagonalLine3DPointX(tmp2D, tmp3D);
			
			const float a = mLineFunction2D[it].a;
			const float b = mLineFunction2D[it].b;
			const float c = mLineFunction2D[it].c;
			mFeatures3D[it].y = -(a*mFeatures3D[it].x+c)/b;
		}
	}
	
}

bool FeaturesPointExtract::BoardLineFunction()
{
	const int Num = mLineFunction2D.size();
	if (Num != mFeaturesNum) return false;
	const float halfInterval = 0.5*mLineInterval;

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
			cv::Point2f p2((it+1)*halfInterval,mLineLength);
			mLineFunction2D[it] = CommonFunctions::ComputeLineFunction2D(p1, p2);
		}
	}
	return true;
}

bool FeaturesPointExtract::Features3DInitialize()
{
	const int Num = mFeatures3D.size();
	if (Num != mFeaturesNum) return false;
	const float halfInterval = 0.5*mLineInterval;

	for (size_t it = 0; it < Num; ++it){
		if (it % 2 == 0) {
			mFeatures3D[it].x = halfInterval*it;
		}
	}
	return true;
}

void FeaturesPointExtract::CalculateFeatures2D()
{
	//运行函数必要的参数
	const int imageRow = mImageHeight;
	const int imageCol = mImageWidth;

	std::vector<std::vector<cv::Point2f>> features;
	for (size_t it = 0; it < mImageHeight; ++it) {
		std::vector<cv::Point2f> featuresPerLine;
		//拿到图像一行像素
		unsigned char* pOneRow = new unsigned char[mImageWidth];
		memcpy(pOneRow, mpFeatureImage, sizeof(unsigned char)*mImageWidth);
		//阈值分割寻找特征区域
		std::vector<std::vector<int>> regions;
		MtxGrowAllSpecifiedRegions grow;
		grow.SetInputData(pOneRow,1,mImageWidth);
		grow.SetLowerLimitVal(0);
		grow.SetUpperLimitVal(128);
		grow.Update();
		regions = grow.GetSpecifiedRegions();
		//灰度质心法求解特征点
		const size_t reionsNum = regions.size();
		for (size_t itRegions = 0; itRegions < reionsNum; ++itRegions) {
			cv::Point2f tmpFeature = CommonFunctions::GrayScaleCentroid(pOneRow, 
				regions[itRegions], 1, mImageWidth);
			featuresPerLine.push_back(tmpFeature);
		}
		if (featuresPerLine.size() != mFeaturesNum) exit(-1);
		features.push_back(featuresPerLine);
		//析构一行图像像素指针
		delete[] pOneRow; pOneRow = nullptr;
	}

	//求解每行同一位置的特征点均值
	for (size_t i = 0; i < mFeaturesNum; ++i) {
		float u = 0., v = 0.;
		for (size_t j = 0; j < mImageHeight; ++j) {
			u += features[j][i].y;
			v += features[j][i].x;
		}
		u /= mImageHeight;
		v /= mImageHeight;
		mFeatures2D[i] = cv::Point2f(u, v);
	}
	return;
}

void FeaturesPointExtract::VerticalLine3DPoints()
{
	if (mFeatures3D.size() != mFeaturesNum) exit(-1);
	std::vector<cv::Point2f> diagonalLinePts;
	diagonalLinePts.reserve(mFeaturesNum);
	for (size_t it = 0; it < mFeaturesNum; ++it) {
		if (it % 2 != 0) {
			cv::Point2f pt(mFeatures3D[it].x, mFeatures3D[it].y);
			diagonalLinePts.push_back(pt);
		}
	}
	CommonStruct::LineFunction2D diagonalLine;
	diagonalLine = CommonFunctions::ComputeLineFunction2D(diagonalLinePts);

	//评定拟合直线的精度
	std::vector<float> errors;
	for (size_t it = 0; it < mFeaturesNum;++it) {
		if (it % 2 != 0) {
			cv::Point2f pt(mFeatures3D[it].x, mFeatures3D[it].y);
			errors.push_back(CommonFunctions::ComputeDistanceFrom2DL2P(diagonalLine, pt));
		}
	}
	float meanErr = CommonFunctions::Mean<float>(errors);
	std::cout << "直线拟合的RMS为：" << meanErr << " mm" << std::endl;

	for (size_t it = 0; it < mFeaturesNum; ++it) {
		if (it % 2 == 0) {
			cv::Point2f pt = CommonFunctions::ComputeIntersectionPt(mLineFunction2D[it], diagonalLine);
			mFeatures3D[it].x = pt.x;
			mFeatures3D[it].y = pt.y;
		}
	}
}

bool FeaturesPointExtract::Update()
{ 
	//初始化
	this->Initialize();
	
	//通过交比不变性计算diagonal line 3D特征点坐标
	this->DiagonalLine3DPoints(mFeatures2D, mFeatures3D);
	
	//通过斜线3D点坐标求解直线3D点坐标
	this->VerticalLine3DPoints();

	return true;
}

bool LineScanCalibration::Update() 
{
	// 判断参数是否正确输入
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