#include "CommonFunctions.h"
#include "LineScanCalibration.h"
#include "GrowAllSpecifiedRegions.h"

#include <fstream>

void FeaturesPointExtract::Initialize()
{
	//获取函数运行必要的参数
	const int featuresNum = this->mFeaturesNum;
	const double lineInterval = this->mLineInterval;
	const double lineLength = this->mLineLength;
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

void FeaturesPointExtract::InitializeWithFeatures()
{
	//获取函数运行必要的参数
	const int featuresNum = this->mFeaturesNum;
	const double lineInterval = this->mLineInterval;
	const double lineLength = this->mLineLength;
	const int imageWidth = this->mImageWidth;
	const int imageHeight = this->mImageHeight;
	const int imageSize = imageWidth*imageHeight;

	//初始化3D点 vertical line上的X坐标
	this->Features3DInitialize();
	//初始化标定板上所有直线的直线方程
	this->BoardLineFunction();
}

bool FeaturesPointExtract::CrossRatio(const std::vector<cv::Point2d>& features, double& crossRatio)
{
	if (features.size() != mCrossRatioPts) return false;
	const double v1 = features[0].y;
	const double v2 = features[1].y;
	const double v3 = features[2].y;
	const double v4 = features[3].y;
	crossRatio = ((v1-v3)*(v2-v4))/((v2-v3)*(v1-v4));
}

bool FeaturesPointExtract::CrossRatio_L(const std::vector<cv::Point2d>& features, double& crossRatio)
{
	if (features.size() != mCrossRatioPts) return false;
	const double v1 = features[0].y;
	const double v2 = features[2].y;
	const double v3 = features[1].y;
	const double v4 = features[3].y;
	crossRatio = ((v1 - v3)*(v2 - v4)) / ((v2 - v3)*(v1 - v4));
}

double FeaturesPointExtract::DiagonalLine3DPointX(const std::vector<cv::Point2d>& features2D,
											    std::vector<cv::Point3d>& features3D)
{
	if (features2D.size() != mCrossRatioPts) exit(-1);
	if (features3D.size() != mCrossRatioPts) exit(-1);

	const double x1 = features3D[0].x;
	const double x2 = features3D[1].x;
	const double x3 = features3D[2].x;
	const double x4 = features3D[3].x;
	
	double crossRatio = 0.;
	this->CrossRatio(features2D, crossRatio);
	double rho = crossRatio*(x1-x4)/(x1-x3);
	return (x4 - rho*x3) / (1 - rho);
}

double FeaturesPointExtract::DiagonalLine3DPointX_L(const std::vector<cv::Point2d>& features2D,
	std::vector<cv::Point3d>& features3D)
{
	if (features2D.size() != mCrossRatioPts) exit(-1);
	if (features3D.size() != mCrossRatioPts) exit(-1);

	const double x1 = features3D[0].x;
	const double x2 = features3D[2].x;
	const double x3 = features3D[1].x;
	const double x4 = features3D[3].x;

	double crossRatio = 0.;
	this->CrossRatio_L(features2D, crossRatio);
	double rho = crossRatio*(x1 - x4) / (x1 - x3);
	return (x4 - rho*x3) / (1 - rho);
}

void FeaturesPointExtract::DiagonalLine3DPoints()
{
	if (mFeatures2D.size() != mFeaturesNum) exit(-1);
	if (mFeatures3D.size() != mFeaturesNum) exit(-1);

	for (int it = 0; it < mFeaturesNum; ++it) {
		//diagonal line
		if (it % 2 != 0) {
			std::vector<cv::Point2d> tmp2D; tmp2D.clear();
			std::vector<cv::Point3d> tmp3D; tmp3D.clear();
			//First Point
			if (it == mFeaturesNum - 2) {
				tmp2D.push_back(mFeatures2D[it - 3]);
				tmp2D.push_back(mFeatures2D[it - 1]);
				tmp2D.push_back(mFeatures2D[it]);
				tmp2D.push_back(mFeatures2D[it + 1]);
			}
			//Normal Point
			else {
				tmp2D.push_back(mFeatures2D[it - 1]);
				tmp2D.push_back(mFeatures2D[it]);
				tmp2D.push_back(mFeatures2D[it + 1]);
				tmp2D.push_back(mFeatures2D[it + 3]);
			}
			if (tmp2D.size() != mCrossRatioPts) exit(-1);
			//First Point
			if (it == mFeaturesNum - 2){
				tmp3D.push_back(mFeatures3D[it - 3]);
				tmp3D.push_back(mFeatures3D[it - 1]);
				tmp3D.push_back(mFeatures3D[it]);
				tmp3D.push_back(mFeatures3D[it + 1]);
			}
			//Normal Point
			else {
				tmp3D.push_back(mFeatures3D[it - 1]);
				tmp3D.push_back(mFeatures3D[it]);
				tmp3D.push_back(mFeatures3D[it + 1]);
				tmp3D.push_back(mFeatures3D[it + 3]);
			}
			if (tmp3D.size() != mCrossRatioPts) exit(-1);

			if (it == mFeaturesNum - 2) {
				mFeatures3D[it].x = this->DiagonalLine3DPointX_L(tmp2D, tmp3D);
			}
			else {
				mFeatures3D[it].x = this->DiagonalLine3DPointX(tmp2D, tmp3D);
			}
			
			const double a = mLineFunction2D[it].a;
			const double b = mLineFunction2D[it].b;
			const double c = mLineFunction2D[it].c;
			mFeatures3D[it].y = -(a*mFeatures3D[it].x+c)/b;
		}
	}
	
}

bool FeaturesPointExtract::BoardLineFunction()
{
	const int Num = mLineFunction2D.size();
	if (Num != mFeaturesNum) return false;
	const double halfInterval = 0.5*mLineInterval;

	for (int it = 0; it < Num; ++it) {
		//vertical line
		if (it % 2 == 0) {
			mLineFunction2D[it].a = 1.0;
			mLineFunction2D[it].b = 0.;
			mLineFunction2D[it].c = -halfInterval*it;
		}
		//diagonal line
		else {
			cv::Point2d p1((it-1)*halfInterval,0.);
			cv::Point2d p2((it+1)*halfInterval,mLineLength);
			mLineFunction2D[it] = CommonFunctions::ComputeLineFunction2D(p1, p2);
		}
	}
	return true;
}

bool FeaturesPointExtract::Features3DInitialize()
{
	const int Num = mFeatures3D.size();
	if (Num != mFeaturesNum) return false;
	const double halfInterval = 0.5*mLineInterval;

	for (int it = 0; it < Num; ++it){
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
	
	std::vector<std::vector<cv::Point2d>> features;
	for (size_t it = 0; it < mImageHeight; ++it) {
		std::vector<cv::Point2d> featuresPerLine;
		//拿到图像一行像素
		unsigned char* pOneRow = new unsigned char[mImageWidth];
		memcpy(pOneRow, &mpFeatureImage[it*mImageWidth], sizeof(unsigned char)*mImageWidth);
		//保存该行图像
		/*std::string file = "C:\\Users\\Administrator\\Desktop\\LineScanCaliData\\tmp.bmp";
		cv::Mat imageTmp = cv::Mat(1, mImageWidth, CV_8UC1, pOneRow);
		cv::imwrite(file, imageTmp);*/
		//阈值分割寻找特征区域
		std::vector<std::vector<int>> regions;
		MtxGrowAllSpecifiedRegions grow;
		grow.SetInputData(pOneRow,mImageWidth,1);
		grow.SetLowerLimitVal(0);
		grow.SetUpperLimitVal(50);
		grow.Update();
		regions = grow.GetSpecifiedRegions();
		//灰度质心法求解特征点
		const size_t reionsNum = regions.size();
		for (size_t itRegions = 0; itRegions < reionsNum; ++itRegions) {
			cv::Point2d tmpFeature = CommonFunctions::GrayScaleCentroid(pOneRow, 
				regions[itRegions], 1, mImageWidth);
			featuresPerLine.push_back(tmpFeature);
		}
		if (featuresPerLine.size() != mFeaturesNum) continue;
		features.push_back(featuresPerLine);
		//析构一行图像像素指针
		delete[] pOneRow; pOneRow = nullptr;
	}
	//求解每行同一位置的特征点均值
	for (int i = 0; i < mFeaturesNum; ++i) {
		double u = 0., v = 0.;
		for (int j = 0; j < features.size(); ++j) {
			u += features[j][i].y;
			v += features[j][i].x;
		}
		u /= mImageHeight;
		v /= mImageHeight;
		mFeatures2D[i] = cv::Point2d(u, v);
	}
	//将特征点绘制在图像上检验特征提取正确性
	cv::Mat imageRaw(imageRow, imageCol, CV_8UC1, mpFeatureImage);
	for (int it = 0; it < mFeaturesNum; ++it) {
		cv::line(imageRaw, cv::Point2d(mFeatures2D[it].y,0),cv::Point2d(mFeatures2D[it].y,1599), cv::Scalar(255, 255, 255), 2, 8, 0);
	}
	std::string saveFile = mDebugPath + "\\features2D.bmp";
	cv::imwrite(saveFile, imageRaw);
	return;
}

void FeaturesPointExtract::GenerateDebugImage()
{
	mImageDebug = cv::Mat(mLineLength, mLineInterval*(mFeaturesNum/2), CV_8UC1, cv::Scalar(255));
	for (int it = 0; it < mFeaturesNum; ++it) {
		//绘制竖直直线
		if (it % 2 == 0) {
			cv::Point2d pt1(it*mLineInterval*0.5,0);
			cv::Point2d pt2(it*mLineInterval*0.5, mLineLength - 1);
			cv::line(mImageDebug,pt1,pt2,cv::Scalar(0,0,0),2,8,0);
		}
		//绘制对角直线
		else {
			cv::Point2d pt1((it-1)*mLineInterval*0.5, 0);
			cv::Point2d pt2((it + 1)*mLineInterval*0.5, mLineLength - 1);
			cv::line(mImageDebug, pt1, pt2, cv::Scalar(0, 0, 0), 2, 8, 0);
		}
	}
	std::string saveFile = mDebugPath + "\\3D.bmp";
	cv::imwrite(saveFile, mImageDebug);
}

void FeaturesPointExtract::VerticalLine3DPoints()
{
	if (mFeatures3D.size() != mFeaturesNum) exit(-1);
	std::vector<cv::Point2d> diagonalLinePts;
	diagonalLinePts.reserve(mFeaturesNum);
	for (size_t it = 0; it < mFeaturesNum; ++it) {
		if (it % 2 != 0) {
			cv::Point2d pt(mFeatures3D[it].x, mFeatures3D[it].y);
			diagonalLinePts.push_back(pt);
		}
	}
	CommonStruct::LineFunction2D diagonalLine;
	diagonalLine = CommonFunctions::ComputeLineFunction2D(diagonalLinePts);
	/*评价拟合直线的精度*/
	std::vector<double> errors;
	for (int it = 0; it < mFeaturesNum;++it) {
		if (it % 2 != 0) {
			cv::Point2d pt(mFeatures3D[it].x, mFeatures3D[it].y);
			errors.push_back(CommonFunctions::ComputeDistanceFrom2DL2P(diagonalLine, pt));
		}
	}
	double meanErr = CommonFunctions::Mean<double>(errors);
	std::cout << "直线拟合的RMS为：" << meanErr << " mm" << std::endl;
	/*计算竖直直线坐标*/
	for (int it = 0; it < mFeaturesNum; ++it) {
		if (it % 2 == 0) {
			cv::Point2d pt = CommonFunctions::ComputeIntersectionPt(mLineFunction2D[it], diagonalLine);
			mFeatures3D[it].x = pt.x;
			mFeatures3D[it].y = pt.y;
		}
	}
}

bool FeaturesPointExtract::Update()
{ 
	/*初始化*/
	this->Initialize();
	/*通过交比不变性计算diagonal line 3D特征点坐标*/
	this->DiagonalLine3DPoints();
	/*通过斜线3D点坐标求解直线3D点坐标*/
	this->VerticalLine3DPoints();
	/*生成debug图像*/
	this->GenerateDebugImage();
	/*正常返回*/
	return true;
}

bool FeaturesPointExtract::UpdateWithFeatures()
{
	/*初始化*/
	this->InitializeWithFeatures();
	/*通过交比不变性计算diagonal line 3D特征点坐标*/
	this->DiagonalLine3DPoints();
	/*通过斜线3D点坐标求解直线3D点坐标*/
	this->VerticalLine3DPoints();
	/*正常返回*/
	return true;
}

bool LineScanCalibration::Update() 
{
	/*加载debug图像*/
	std::string fileLoad = mDebugPath + "\\3D.bmp";
	mImageDebug = cv::imread(fileLoad, cv::IMREAD_GRAYSCALE);
	/*判断参数是否正确输入*/
	if (mObjectPoints.empty() || mImagePoints.empty()) {
		std::cerr << "Object Points || Image Points Size Empty..." << std::endl;
		return false;
	}
	if (mObjectPoints.size() != mImagePoints.size()) {
		std::cerr << "Object Points's size not equal to Image Points's..." << std::endl;
		return false;
	}
	/*初始化标定参数*/
	if (!this->InitialEstimate3()) {
		std::cerr << "Initial Estimate Failed..." << std::endl;
		return false;
	}
	/*保存初始化标定结果*/
	this->SaveWorldPointsBeforeOptimized(mIniCameraPara[0]);
	/*优化标定参数*/
	if (!this->OptimizeEstimate()) {
		std::cerr << "Optimize Estimate Failed..." << std::endl;
		return false;
	}
	/*计算标定图像分辨率*/
	if (!this->Resolution()) {
		std::cerr << "Resolution Estimate Failed..." << std::endl;
		return false;
	}
	/*保存相关标定信息*/
	this->SaveWorldPointsAfterOptimized(mCameraPara);
	this->SaveDebugImage();
	this->SaveGroundTruth();
	/*正常返回*/
	return true;
}

bool LineScanCalibration::InitialEstimate3()
{
	//输出调试信息
	CommonFunctions::ConditionPrint("Start Initial Estimate...");
	//函数运行前必进行的检查工作
	if (mObjectPoints.size() != mImagePoints.size()) {
		std::cerr << "Object Points's Are Not Equal To Image Points's" << std::endl;
		return false;
	}
	if (mObjectPoints.empty() || mImagePoints.empty()) {
		std::cerr << "Features Points Is Empty" << std::endl;
		return false;
	}
	const int imagesNum = mObjectPoints.size();
	int featuresNum = mObjectPoints.begin()->size();
	//初始化参数矩阵
	Eigen::MatrixXd C1(featuresNum*imagesNum, 3);
	Eigen::MatrixXd C2(featuresNum*imagesNum, 3);
	Eigen::MatrixXd C3(featuresNum*imagesNum, 2);
	Eigen::Vector3d sumPts = Eigen::Vector3d::Zero();
	//根据2D点与3D点的坐标，估计投影矩阵M
	for (int i = 0; i < imagesNum; ++i) {
		for (int j = 0; j < featuresNum; ++j) {
			/*构造C1*/
			C1(j + featuresNum*i, 0) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].x));
			C1(j + featuresNum*i, 1) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].y));
			C1(j + featuresNum*i, 2) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].z));
			/*构造C2*/
			C2(j + featuresNum*i, 0) = double(mObjectPoints[i][j].x);
			C2(j + featuresNum*i, 1) = double(mObjectPoints[i][j].y);
			C2(j + featuresNum*i, 2) = double(mObjectPoints[i][j].z);
			/*构造C3*/
			C3(j + featuresNum*i, 0) = 1.0;
			C3(j + featuresNum*i, 1) = double(-(mImagePoints[i][j].y));
			/*参数求和*/
			sumPts(0, 0) += double(mObjectPoints[i][j].x);
			sumPts(1, 0) += double(mObjectPoints[i][j].y);
			sumPts(2, 0) += double(mObjectPoints[i][j].z);
		}
	}
	sumPts /= double(featuresNum*imagesNum);
	/*矩阵运算*/
	Eigen::MatrixXd C1T = C1.transpose();
	Eigen::MatrixXd C2T = C2.transpose();
	Eigen::MatrixXd C3T = C3.transpose();
	Eigen::MatrixXd C1TC1 = C1T*C1;
	Eigen::MatrixXd C1TC2 = C1T*C2;
	Eigen::MatrixXd C1TC3 = C1T*C3;
	Eigen::MatrixXd C2TC1 = C2T*C1;
	Eigen::MatrixXd C2TC2 = C2T*C2;
	Eigen::MatrixXd C2TC3 = C2T*C3;
	Eigen::MatrixXd C3TC1 = C3T*C1;
	Eigen::MatrixXd C3TC2 = C3T*C2;
	Eigen::MatrixXd C3TC3 = C3T*C3;
	Eigen::MatrixXd C2TC2_inv = C2TC2.inverse();
	/*计算系数矩阵*/
	Eigen::MatrixXd factor = (C3TC2*C2TC2_inv*C2TC3 - C3TC3);
	Eigen::MatrixXd A = factor.inverse()*(C3TC1 - C3TC2*C2TC2_inv*C2TC1);
	Eigen::MatrixXd B = -C2TC2_inv*(C2TC1 + C2TC3*A);
	Eigen::MatrixXd D = -(C1TC1 + C1TC2*B + C1TC3*A);
	/*std::cout << "C2" << std::endl;
	std::cout << C2 << std::endl;
	std::cout << "C3" << std::endl;
	std::cout << C3 << std::endl;
	std::cout << "C2TC3" << std::endl;
	std::cout << C2TC3 << std::endl;
	std::cout << "C3TC2" << std::endl;
	std::cout << C3TC2 << std::endl;
	std::cout << "C3TC3" << std::endl;
	std::cout << C3TC3 << std::endl;
	std::cout << "factor:" << std::endl;
	std::cout << factor << std::endl;
	std::cout << factor.inverse() << std::endl;
	std::cout << factor*factor.inverse() << std::endl;
	std::cout << "C2TC2" << std::endl;
	std::cout << C2TC2 << std::endl;
	std::cout << C2TC2.inverse() << std::endl;
	std::cout << C2TC2*C2TC2.inverse() << std::endl;
	std::cout << "A" << std::endl;
	std::cout << A << std::endl;
	std::cout << "B" << std::endl;
	std::cout << B << std::endl;*/
	/*取重投影误差最小的特征向量*/
	Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(D);
	/*Eigen::MatrixXcd eigenvectors = es.eigenvectors();
	Eigen::MatrixXcd eigenvalues = es.eigenvalues();*/
	std::cout << "eigenvalues: " << std::endl << es.eigenvalues() << std::endl;
	std::cout << "eigenvectors: " << std::endl << es.eigenvectors() << std::endl;
	float minReproError = FLT_MAX;
	Eigen::Matrix<double, 2, 4> Mrepro;
	Eigen::MatrixXd phi1 = Eigen::MatrixXd(3, 1);
	Eigen::MatrixXd phi2 = Eigen::MatrixXd(3, 1);
	Eigen::MatrixXd phi3 = Eigen::MatrixXd(2, 1);
	for (int it = 0;it < 3;++it) {
		/*验证特征值与特征向量计算是否正确*/
		Eigen::MatrixXd Check1 = D*es.eigenvectors().real().col(it);
		Eigen::MatrixXd Check2 = es.eigenvectors().real().col(it)*es.eigenvalues().real().row(it);
		/*std::cout << "Check1: " << std::endl << Check1 << std::endl;
		std::cout << "Check2: " << std::endl << Check2 << std::endl;*/
		phi1(0, 0) = static_cast<double>(es.eigenvectors().real()(0, it));
		phi1(1, 0) = static_cast<double>(es.eigenvectors().real()(1, it));
		phi1(2, 0) = static_cast<double>(es.eigenvectors().real()(2, it));
		phi2 = B*phi1;
		phi3 = A*phi1;
		/*if (phi5(3, 0) < 0) {
		phi3 = -phi3;
		phi5 = -(C5TC5).inverse()*C5T*C3*phi3;;
		}*/
		/*std::cout << phi1 << std::endl;
		std::cout << phi2 << std::endl;
		std::cout << phi3 << std::endl;*/
		/*特征值组成投影矩阵*/
		double m24 = phi3(0, 0);
		double m34 = phi3(1, 0);
		Eigen::MatrixXd m3 = phi1.transpose();
		Eigen::MatrixXd m2 = phi2.transpose();
		/*估计当前M矩阵的重投影误差*/
		Eigen::Matrix<double, 2, 4> Mtmp;
		Mtmp.block(0, 0, 1, 3) = m2.block(0, 0, 1, 3);Mtmp(0, 3) = m24;
		Mtmp.block(1, 0, 1, 3) = m3.block(0, 0, 1, 3);Mtmp(1, 3) = m34;
		/*std::cout << Mtmp << std::endl;*/
		/*计算相机内外参数*/
		InstrinsticPara instrinstictmp;
		ExtrinsticPara extrinstictmp;
		this->ReshapePara(Mtmp, sumPts, instrinstictmp, extrinstictmp);
		/*计算重投影误差*/
		double reproError = 0.;
		Eigen::Vector3d distort = Eigen::Vector3d::Zero();
		for (int it = 0; it < imagesNum; ++it) {
			/*double reproError = this->ReprojectError(mImagePoints[0], mObjectPoints[0], Mtmp);*/
			reproError += this->ReprojectError(mImagePoints[it], mObjectPoints[it],
				extrinstictmp.R, extrinstictmp.T,distort,instrinstictmp.Fy, instrinstictmp.vc);
		}
		reproError /= imagesNum;
		/*选择最小重投影误差对应的M矩阵*/
		if (reproError < minReproError && m34 > 0) {
			Mrepro = Mtmp;
			minReproError = reproError;
		}
	}
	/*打印重投影误差*/
	std::cout << "Reproject Error Before Optimize: " << minReproError << std::endl;
	/*计算相机内外参数*/
	InstrinsticPara instrinstic;
	ExtrinsticPara extrinstic;
	this->ReshapePara(Mrepro, sumPts, instrinstic, extrinstic);
	/*记录初始相机内外参数*/
	LineScanPara cameraPara;
	cameraPara.Fy = instrinstic.Fy;
	cameraPara.vc = instrinstic.vc;
	cameraPara.R = extrinstic.R;
	cameraPara.T = extrinstic.T;
	mIniCameraPara.push_back(cameraPara);
	//输出调试信息
	CommonFunctions::ConditionPrint("End Initial Estimate");
	return true;
}

bool LineScanCalibration::InitialEstimate2()
{
	//输出调试信息
	CommonFunctions::ConditionPrint("Start Initial Estimate...");
	//函数运行前必进行的检查工作
	if (mObjectPoints.size() != mImagePoints.size()) {
		std::cerr << "Object Points's Are Not Equal To Image Points's" << std::endl;
		return false;
	}
	if (mObjectPoints.empty() || mImagePoints.empty()) {
		std::cerr << "Features Points Is Empty" << std::endl;
		return false;
	}
	//根据2D点与3D点的坐标，估计投影矩阵M
	const int imagesNum = mObjectPoints.size();
	int featuresNum = mObjectPoints.begin()->size();
	for (int i = 0; i < imagesNum; ++i) {
		Eigen::MatrixXd C1(featuresNum, 3);
		Eigen::MatrixXd C2(featuresNum, 3);
		Eigen::MatrixXd C3(featuresNum, 2);
		Eigen::Vector3d sumPts = Eigen::Vector3d::Zero();
		/*保存系数矩阵*/
		/*std::ofstream outfile;
		outfile.open("C:\\Users\\Public\\Desktop\\matrix.txt", std::ofstream::app);
		std::ofstream outfile2D;
		outfile2D.open("C:\\Users\\Public\\Desktop\\2D.txt", std::ofstream::app);
		std::ofstream outfile3D;
		outfile3D.open("C:\\Users\\Public\\Desktop\\3D.txt", std::ofstream::app);*/
		for (int j = 0; j < featuresNum; ++j) {
			/*构造C1*/
			C1(j, 0) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].x));
			C1(j, 1) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].y));
			C1(j, 2) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].z));
			/*构造C2*/
			C2(j, 0) = double(mObjectPoints[i][j].x);
			C2(j, 1) = double(mObjectPoints[i][j].y);
			C2(j, 2) = double(mObjectPoints[i][j].z);
			/*构造C3*/
			C3(j, 0) = 1000.0;
			C3(j, 1) = double(-(mImagePoints[i][j].y));
			/*记录系数矩阵*/
			/*outfile << C2(j, 0) << "\t" << C2(j, 1) << "\t" << C2(j, 2) << "\t" << C3(j, 0) << "\t"
				<< C1(j, 0) << "\t" << C1(j, 1) << "\t" << C1(j, 2) << "\t" << C3(j, 1) << std::endl; */
			/*记录特征点*/
			/*outfile2D << mImagePoints[i][j].x << "\t" << mImagePoints[i][j].y << std::endl;
			outfile3D << mObjectPoints[i][j].x << "\t" << mObjectPoints[i][j].y << "\t" << mObjectPoints[i][j].z << std::endl;*/
			/*参数求和*/
			sumPts(0, 0) += double(mObjectPoints[i][j].x);
			sumPts(1, 0) += double(mObjectPoints[i][j].y);
			sumPts(2, 0) += double(mObjectPoints[i][j].z);
		}
		/*outfile2D << std::endl << "--------------------------------" << std::endl << std::endl;
		outfile3D << std::endl << "--------------------------------" << std::endl << std::endl;
		outfile.close();
		outfile2D.close();
		outfile3D.close();*/
		sumPts /= double(featuresNum);
		/*矩阵运算*/
		Eigen::MatrixXd C1T = C1.transpose();
		Eigen::MatrixXd C2T = C2.transpose();
		Eigen::MatrixXd C3T = C3.transpose();
		Eigen::MatrixXd C1TC1 = C1T*C1;
		Eigen::MatrixXd C1TC2 = C1T*C2;
		Eigen::MatrixXd C1TC3 = C1T*C3;
		Eigen::MatrixXd C2TC1 = C2T*C1;
		Eigen::MatrixXd C2TC2 = C2T*C2;
		Eigen::MatrixXd C2TC3 = C2T*C3;
		Eigen::MatrixXd C3TC1 = C3T*C1;
		Eigen::MatrixXd C3TC2 = C3T*C2;
		Eigen::MatrixXd C3TC3 = C3T*C3;
		Eigen::MatrixXd C2TC2_inv = C2TC2.inverse();
		/*计算系数矩阵*/
		Eigen::MatrixXd factor = (C3TC2*C2TC2_inv*C2TC3 - C3TC3);
		Eigen::MatrixXd A = factor.inverse()*(C3TC1 - C3TC2*C2TC2_inv*C2TC1);
		Eigen::MatrixXd B = -C2TC2_inv*(C2TC1 + C2TC3*A);
		Eigen::MatrixXd D = -(C1TC1 + C1TC2*B + C1TC3*A);
		/*std::cout << "C2TC3" << std::endl;
		std::cout << C2TC3 << std::endl;
		std::cout << "C3TC2" << std::endl;
		std::cout << C3TC2 << std::endl;
		std::cout << "C3TC3" << std::endl;
		std::cout << C3TC3 << std::endl;
		std::cout << "factor:" << std::endl;
		std::cout << factor << std::endl;
		std::cout << factor.inverse() << std::endl;
		std::cout << factor*factor.inverse() << std::endl;
		std::cout << "C2TC2" << std::endl;
		std::cout << C2TC2 << std::endl;
		std::cout << C2TC2.inverse() << std::endl;
		std::cout << C2TC2*C2TC2.inverse() << std::endl;
		std::cout << "A" << std::endl;
		std::cout << A << std::endl;
		std::cout << "B" << std::endl;
		std::cout << B << std::endl;*/
		/*取重投影误差最小的特征向量*/
		Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(D);
		/*Eigen::MatrixXcd eigenvectors = es.eigenvectors();
		Eigen::MatrixXcd eigenvalues = es.eigenvalues();*/
		std::cout << "eigenvalues: " << std::endl << es.eigenvalues() << std::endl;
		std::cout << "eigenvectors: " << std::endl << es.eigenvectors() << std::endl;
		float minReproError = FLT_MAX;
		Eigen::Matrix<double, 2, 4> Mrepro;
		Eigen::MatrixXd phi1 = Eigen::MatrixXd(3, 1);
		Eigen::MatrixXd phi2 = Eigen::MatrixXd(3, 1);
		Eigen::MatrixXd phi3 = Eigen::MatrixXd(2, 1);
		for (int it = 0;it < 3;++it) {
			/*验证特征值与特征向量计算是否正确*/
			Eigen::MatrixXd Check1 = D*es.eigenvectors().real().col(it);
			Eigen::MatrixXd Check2 = es.eigenvectors().real().col(it)*es.eigenvalues().real().row(it);
			/*std::cout << "Check1: " << std::endl << Check1 << std::endl;
			std::cout << "Check2: " << std::endl << Check2 << std::endl;*/
			phi1(0, 0) = static_cast<double>(es.eigenvectors().real()(0, it));
			phi1(1, 0) = static_cast<double>(es.eigenvectors().real()(1, it));
			phi1(2, 0) = static_cast<double>(es.eigenvectors().real()(2, it));
			phi2 = B*phi1;
			phi3 = A*phi1;
			/*if (phi5(3, 0) < 0) {
			phi3 = -phi3;
			phi5 = -(C5TC5).inverse()*C5T*C3*phi3;;
			}*/
			/*std::cout << phi1 << std::endl;
			std::cout << phi2 << std::endl;
			std::cout << phi3 << std::endl;*/
			/*特征值组成投影矩阵*/
			double m24 = phi3(0, 0);
			double m34 = phi3(1, 0);
			Eigen::MatrixXd m3 = phi1.transpose();
			Eigen::MatrixXd m2 = phi2.transpose();
			/*估计当前M矩阵的重投影误差*/
			Eigen::Matrix<double, 2, 4> Mtmp;
			Mtmp.block(0, 0, 1, 3) = m2.block(0, 0, 1, 3);Mtmp(0, 3) = m24*1000;
			Mtmp.block(1, 0, 1, 3) = m3.block(0, 0, 1, 3);Mtmp(1, 3) = m34;
			std::cout << Mtmp << std::endl;
			double reproError = this->ReprojectError(mImagePoints[i], mObjectPoints[i], Mtmp);
			/*计算相机内外参数*/
			InstrinsticPara instrinstictmp;
			ExtrinsticPara extrinstictmp;
			this->ReshapePara(Mtmp, sumPts, instrinstictmp, extrinstictmp);
			Eigen::Vector3d dis = Eigen::Vector3d::Zero();
			double reproError2 = this->ReprojectError(mImagePoints[i], mObjectPoints[i],
				extrinstictmp.R, extrinstictmp.T, dis, instrinstictmp.Fy, instrinstictmp.vc);
			
			if (reproError2 < minReproError /*&& m34 > 0*/) {
				Mrepro = Mtmp;
				minReproError = reproError2;
			}
		}
		/*打印重投影误差*/
		std::cout << "Reproject Error Before Optimize: " << minReproError << std::endl;
		/*计算相机内外参数*/
		InstrinsticPara instrinstic;
		ExtrinsticPara extrinstic;
		this->ReshapePara(Mrepro, sumPts, instrinstic, extrinstic);
		/*记录初始相机内外参数*/
		LineScanPara cameraPara;
		cameraPara.Fy = instrinstic.Fy;
		cameraPara.vc = instrinstic.vc;
		cameraPara.R = extrinstic.R;
		cameraPara.T = extrinstic.T;
		mIniCameraPara.push_back(cameraPara);
	}
	//输出调试信息
	CommonFunctions::ConditionPrint("End Initial Estimate");
	return true;
}

bool LineScanCalibration::InitialEstimate() 
{
	//struct FrameInfo {
	//	Eigen::Matrix3f R;
	//	Eigen::Vector3f T;
	//	float Fy;
	//	float vc;
	//};
	////输出调试信息
	//CommonFunctions::ConditionPrint("Start Initial Estimate...");	
	////函数运行前必进行的检查工作
	//if (mObjectPoints.size() != mImagePoints.size()){
	//	std::cerr << "Object Points's Are Not Equal To Image Points's" << std::endl;
	//	return false;
	//}
	//if (mObjectPoints.empty() || mImagePoints.empty()) {
	//	std::cerr << "Features Points Is Empty" << std::endl;
	//	return false;
	//}
	////根据2D点与3D点的坐标，估计投影矩阵M
	//const int imagesNum = mObjectPoints.size();
	//int featuresNum = mObjectPoints.begin()->size();
	//for (int i = 0; i < imagesNum; ++i) {
	//	Eigen::MatrixXd C3(featuresNum, 3);
	//	Eigen::MatrixXd C5(featuresNum, 4);
	//	Eigen::MatrixXd C8(featuresNum, 8);
	//	Eigen::Vector3f sumPts = Eigen::Vector3f::Zero();
	//	for (int j = 0; j < featuresNum; ++j) {
	//		C3(j, 0) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].x));
	//		C3(j, 1) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].y));
	//		C3(j, 2) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].z));
	//		
	//		C5(j, 0) = double(mObjectPoints[i][j].x);
	//		C5(j, 1) = double(mObjectPoints[i][j].y);
	//		C5(j, 2) = double(mObjectPoints[i][j].z);
	//		//C5(j, 3) = 1.0;
	//		C5(j, 3) = double(mImagePoints[i][j].y);

	//		C8(j, 0) = double(mObjectPoints[i][j].x);
	//		C8(j, 1) = double(mObjectPoints[i][j].y);
	//		C8(j, 2) = double(mObjectPoints[i][j].z);
	//		C8(j, 3) = 1.0;
	//		C8(j, 4) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].x));
	//		C8(j, 5) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].y));
	//		C8(j, 6) = double(-(mImagePoints[i][j].y)*(mObjectPoints[i][j].z));
	//		C8(j, 7) = double(mImagePoints[i][j].y);

	//		sumPts(0, 0) += float(mObjectPoints[i][j].x);
	//		sumPts(1, 0) += float(mObjectPoints[i][j].y);
	//		sumPts(2, 0) += float(mObjectPoints[i][j].z);
	//	}
	//	/*计算法方程D*/
	//	/*sumPts /= float(featuresNum);*/
	//	//C5 /= 1000.;
	//	Eigen::MatrixXd C3T = C3.transpose();
	//	Eigen::MatrixXd C5T = C5.transpose();
	//	Eigen::MatrixXd C5TC5 = C5T*C5;
	//	/*std::cout << "C5" << std::endl;
	//	std::cout << C5 << std::endl;
	//	std::cout << C5TC5 << std::endl;
	//	std::cout << C5TC5.inverse() << std::endl;
	//	std::cout << C5TC5*C5TC5.inverse() << std::endl;*/
	//	Eigen::MatrixXd D = -C3T*C3 + C3T*C5*(C5TC5).inverse()*C5T*C3;
	//	/*std::cout << "D" << std::endl;
	//	std::cout << D << std::endl;
	//	std::cout << D.inverse() << std::endl;
	//	std::cout << D*D.inverse() << std::endl;*/
	//	/*取重投影误差最小的特征向量*/
	//	Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(D);
	//	/*Eigen::MatrixXcd eigenvectors = es.eigenvectors();
	//	Eigen::MatrixXcd eigenvalues = es.eigenvalues();*/
	//	/*std::cout << "eigenvalues: " << std::endl << es.eigenvalues() << std::endl;
	//	std::cout << "eigenvectors: " << std::endl << es.eigenvectors() << std::endl;*/

	//	Eigen::EigenSolver<Eigen::Matrix<double, 8, 8>> es2(C8.transpose()*C8);
	//	/*std::cout << "eigenvalues2: " << std::endl << es2.eigenvalues() << std::endl;
	//	std::cout << "eigenvectors2: " << std::endl << es2.eigenvectors() << std::endl;*/

	//	/*Eigen::MatrixXd eigenvalues(3,3);
	//	Eigen::MatrixXd eigenvectors(3,3);
	//	this->EigenSVD(D, eigenvalues, eigenvectors);*/
	//	/*Eigen::MatrixXd evalsReal = evals.real();
	//	Eigen::MatrixXf::Index evalsMin;
	//	evalsReal.rowwise().sum().minCoeff(&evalsMin);*/
	//	FrameInfo frameInfo;
	//	float minReproError = FLT_MAX;
	//	Eigen::Matrix<float, 2, 4> Mrepro;
	//	Eigen::MatrixXd phi3 = Eigen::MatrixXd(3, 1);
	//	Eigen::MatrixXd phi5 = Eigen::MatrixXd(4, 1);
	//	for (int it = 0;it < 3;++it) {
	//		/*验证特征值与特征向量计算是否正确*/
	//		Eigen::MatrixXd Check1 = D*es.eigenvectors().real().col(it);
	//		Eigen::MatrixXd Check2 = es.eigenvectors().real().col(it)*es.eigenvalues().real().row(it);
	//		/*std::cout << "Check1: " << std::endl << Check1 << std::endl;
	//		std::cout << "Check2: " << std::endl << Check2 << std::endl;*/
	//		phi3(0, 0) = static_cast<double>(es.eigenvectors().real()(0, it));
	//		phi3(1, 0) = static_cast<double>(es.eigenvectors().real()(1, it));
	//		phi3(2, 0) = static_cast<double>(es.eigenvectors().real()(2, it));
	//		phi5 = -(C5TC5).inverse()*C5T*C3*phi3;
	//		/*if (phi5(3, 0) < 0) {
	//			phi3 = -phi3;
	//			phi5 = -(C5TC5).inverse()*C5T*C3*phi3;;
	//		}*/
	//		//std::cout << phi3 << std::endl;
	//		//std::cout << phi5 << std::endl;
	//		/*特征值组成投影矩阵*/
	//		float m24 = 0.0;
	//		float m34 = phi5(3, 0);
	//		Eigen::MatrixXd m3 = phi3.transpose();
	//		Eigen::MatrixXd m2 = phi5.block(0, 0, 3, 1).transpose();
	//		/*估计当前M矩阵的重投影误差*/
	//		Eigen::Matrix<float, 2, 4> Mtmp;
	//		Mtmp << float(m2(0, 0)), float(m2(0, 1)), float(m2(0, 2)), float(m24),
	//			float(m3(0, 0)), float(m3(0, 1)), float(m3(0, 2)), float(m34);
	//		float reproError = this->ReprojectError(mImagePoints[i], mObjectPoints[i], Mtmp);
	//		std::cout << "reproject Error: " << reproError << std::endl;
	//		if (reproError < minReproError) {
	//			Mrepro = Mtmp;
	//			minReproError = reproError;
	//		}
	//	}
	//	/*计算相机内外参数*/
	//	InstrinsticPara instrinstic;
	//	ExtrinsticPara extrinstic;
	//	this->ReshapePara(Mrepro, sumPts, instrinstic, extrinstic);
	//	/*记录初始相机内外参数*/
	//	LineScanPara cameraPara;
	//	cameraPara.Fy = instrinstic.Fy;
	//	cameraPara.vc = instrinstic.vc;
	//	cameraPara.R.push_back(extrinstic.R);
	//	cameraPara.T.push_back(extrinstic.T);
	//	mIniCameraPara.push_back(cameraPara);
	//}
	//
	////输出调试信息
	//CommonFunctions::ConditionPrint("End Initial Estimate");
	return true;
}

void LineScanCalibration::ReshapePara(const Eigen::Matrix<double, 2, 4>& M, const Eigen::Vector3d& sumPts,InstrinsticPara& paraIn, ExtrinsticPara& paraEx)
{
	/*获取参数*/
	double m24 = M(0, 3);
	double m34 = M(1, 3);
	Eigen::Matrix<double, 1, 3> m2 = M.block(0, 0, 1, 3);                                
	Eigen::Matrix<double, 1, 3> m3 = M.block(1, 0, 1, 3);
	/*计算相机内参*/
	paraIn.vc = m2(0,0)*m3(0,0)+m2(0,1)*m3(0,1)+m2(0,2)*m3(0,2);
	paraIn.Fy = (m2*skew(m3)).norm();
	/*计算相机外参*/
	Eigen::Matrix<double,1,3> r2 = (m2 - paraIn.vc*m3) / paraIn.Fy;
	Eigen::Matrix<double,1,3> r3 = m3;
	Eigen::Matrix<double,1,3> r1 = (r2*skew(r3));
	paraEx.R.block(0, 0, 1, 3) = r1/r1.norm();
	paraEx.R.block(1, 0, 1, 3) = r2/r2.norm();
	paraEx.R.block(2, 0, 1, 3) = r3/r3.norm();
	paraEx.T(0, 0) = -r1*sumPts;
	paraEx.T(1, 0) = (m24 - paraIn.vc*m34) / paraIn.Fy;
	paraEx.T(2, 0) = m34;
	/*正常退出*/
	return;
}

bool LineScanCalibration::OptimizeEstimate() 
{
	CommonFunctions::ConditionPrint("Start Optimize Estimate...");
	//函数运行前必进行的检查工作
	if (mObjectPoints.size() != mImagePoints.size()) {
		std::cerr << "Object Points's Are Not Equal To Image Points's" << std::endl;
		return false;
	}
	if (mObjectPoints.empty() || mImagePoints.empty()) {
		std::cerr << "Features Points Is Empty" << std::endl;
		return false;
	}
	//获取待优化的参数
	Eigen::Vector3d r;
	Eigen::Vector3d t;
	const int imagesNum = mObjectPoints.size();
	const int featuresNum = mObjectPoints.begin()->size();
	t = mIniCameraPara[0].T;
	r = RotationMatrix2Vector(mIniCameraPara[0].R);
	mCameraPara.Fy = mIniCameraPara[0].Fy;
	mCameraPara.vc = mIniCameraPara[0].vc;
	//待优化的相机内参以及畸变参数
	ceres::Problem problem;
	double k[5] = { mCameraPara.Fy,mCameraPara.vc,mCameraPara.k1,mCameraPara.k2,mCameraPara.k3 };
	for (int i = 0; i < imagesNum; ++i) {
		for (int j = 0; j < featuresNum; ++j) {
			ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<LineScanProjectCost, 2, 5, 3, 3>(
				new LineScanProjectCost(mObjectPoints[i][j], mImagePoints[i][j]));
			problem.AddResidualBlock(cost, nullptr, k, r.data(), t.data());
		}
	}
	std::cout << "Start Linear Optimize..." << std::endl;
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
	options.preconditioner_type = ceres::JACOBI;
	options.max_num_iterations = 2000;
	options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
	options.function_tolerance = 1e-16;
	options.gradient_tolerance = 1e-10;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	/*打印优化的关键信息*/
	if (!summary.IsSolutionUsable()) {
		std::cerr << "Nonlinear Optimization Failed..." << std::endl;
	}
	else {
		std::cout << std::endl
			<< " Bundle Adjustment Statistics (Approximated RMSE):\n"
			<< " #views: " << imagesNum << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " #num_parameters: " << summary.num_parameters << "\n"
			<< " #num_parameter_blocks: " << summary.num_parameter_blocks << "\n"
			<< " #Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " #Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " #Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
		/*确定线扫相机内参*/
		mCameraPara.Fy = k[0];
		mCameraPara.vc = k[1];
		mCameraPara.k1 = k[2];
		mCameraPara.k2 = k[3];
		mCameraPara.k3 = k[4];
		/*确定线扫相机外参*/
		mCameraPara.T = t;
		mCameraPara.R = this->RotationVector2Matrix(r);
		/*打印重投影误差*/
		Eigen::Vector3d distort;
		distort << mCameraPara.k1, mCameraPara.k2, mCameraPara.k3;
		for (int it = 0; it < imagesNum; ++it) {
			double reproError = this->ReprojectError(mImagePoints[it], mObjectPoints[it],
				mCameraPara.R, mCameraPara.T, distort, mCameraPara.Fy, mCameraPara.vc);
			std::cout << "Reprojection Error After Optimize: " << reproError << std::endl;
		}
	}
	CommonFunctions::ConditionPrint("End Optimize Estimate");
	return true;
}

bool LineScanCalibration::Resolution()
{
	/*获取相机标定参数*/
	const double k1 = mCameraPara.k1;
	const double k2 = mCameraPara.k2;
	const double k3 = mCameraPara.k3;
	const double fx = 1.0;
	const double fy = mCameraPara.Fy;
	const double cx = 0.0;
	const double cy = mCameraPara.vc;
	const int imageSize = mImagePoints.size();
	const int featuresSize = mImagePoints.begin()->size();
	for (int i = 0; i < imageSize; ++i) {
		std::vector<cv::Point2d> imagePoints;
		for (int j = 0; j < featuresSize; ++j) {
			/*计算归一化相机坐标系的坐标*/
			cv::Point3d pt;
			pt.z = 1.0;
			pt.x = mImagePoints[i][j].x;
			pt.y = (mImagePoints[i][j].y - cy) / fy;
			/*去除特征点畸变*/
			double xdis = 0., ydis = 0.;
			xdis = pt.x;
			ydis = pt.y - (k1*pt.y*pt.y*pt.y + k2*pt.y*pt.y*pt.y*pt.y*pt.y + k3*pt.y*pt.y);
			/*归一化相机坐标转化为像素坐标*/
			cv::Point2d ptDeDis;
			ptDeDis.x = xdis;
			ptDeDis.y = fy*ydis + cy;
			imagePoints.push_back(ptDeDis);
		}
		mImagePointsDeDis.push_back(imagePoints);
	}
	/*通过交比不变性重新构造3D特征点*/
	for (int it = 0; it < imageSize; ++it) {
		float featuresHeight = mObjectPoints[it].begin()->z;
		std::vector<cv::Point3d> features3D;
		FeaturesPointExtract featuresExtract(mImagePointsDeDis[it], featuresSize, featuresHeight);
		featuresExtract.SetCalibrationPara(10.0, 40.0);//mm
		featuresExtract.UpdateWithFeatures();
		featuresExtract.Get3DPoints(features3D);
		mObjectPointsDeDis.push_back(features3D);
	}
	/*计算相机的分辨率*/
	std::vector<std::vector<double>> resolutions;
	for (int i = 0; i < imageSize; ++i) {
		std::vector<double> resolution;
		for (int j = 0; j < featuresSize - 1; ++j) {
			int idCur = j;int idNext = j + 1;
			const cv::Point2d p12D = mImagePointsDeDis[i][idCur];
			const cv::Point2d p22D = mImagePointsDeDis[i][idNext];
			const cv::Point3d p13D = mObjectPointsDeDis[i][idCur];
			const cv::Point3d p23D = mObjectPointsDeDis[i][idNext];
			double dis2D = CommonFunctions::ComputeDistanceP2P(p12D, p22D);
			double dis3D = CommonFunctions::ComputeDistanceP2P(p13D, p23D);
			resolution.push_back(dis3D / dis2D);
		}
		resolutions.push_back(resolution);
	}
	/*判断分辨率的置信度，以1微米为标准*/
	double threshold = 0.001;
	mCameraPara.Conf = true;
	std::vector<double> resolution;
	for (int i = 0; i < imageSize; ++i) {
		if (!resolutions.empty()) {
			std::sort(resolutions[i].begin(), resolutions[i].end());
			int start = 0; int end = resolutions[i].size() - 1;
			if (abs(resolutions[i][start] - resolutions[i][end]) > threshold) {
				mCameraPara.Conf = false;
			}
			resolution.push_back(CommonFunctions::Average(resolutions[i]));
		}
	}
	mCameraPara.resY = CommonFunctions::Average(resolution);
	/*正常返回*/
	return true;
}

Eigen::MatrixXd LineScanCalibration::skew(Eigen::MatrixXd& vec)
{
	const double x = vec(0,0);
	const double y = vec(0,1);
	const double z = vec(0,2);
	Eigen::MatrixXd res(3, 3);
	res(0, 0) = 0; res(0, 1) = -z; res(0, 2) = y;
	res(1, 0) = z; res(1, 1) = 0; res(1, 2) = -x;
	res(2, 0) = -y; res(2, 1) = x; res(2, 2) = 0;
	return res;
}

Eigen::Matrix3f LineScanCalibration::skew(Eigen::Matrix<float,1,3>& vec)
{
	const double x = vec(0, 0);
	const double y = vec(0, 1);
	const double z = vec(0, 2);
	Eigen::Matrix3f res(3, 3);
	res(0, 0) = 0; res(0, 1) = -z; res(0, 2) = y;
	res(1, 0) = z; res(1, 1) = 0; res(1, 2) = -x;
	res(2, 0) = -y; res(2, 1) = x; res(2, 2) = 0;
	return res;
}

Eigen::Matrix3d LineScanCalibration::skew(Eigen::Matrix<double, 1, 3>& vec)
{
	const double x = vec(0, 0);
	const double y = vec(0, 1);
	const double z = vec(0, 2);
	Eigen::Matrix3d res(3, 3);
	res(0, 0) = 0; res(0, 1) = -z; res(0, 2) = y;
	res(1, 0) = z; res(1, 1) = 0; res(1, 2) = -x;
	res(2, 0) = -y; res(2, 1) = x; res(2, 2) = 0;
	return res;
}

Eigen::Vector3d LineScanCalibration::RotationMatrix2Vector(const Eigen::Matrix3d& R)
{
	Eigen::AngleAxisd r;
	r.fromRotationMatrix(R);
	return r.angle()*r.axis();
}

Eigen::Matrix3d LineScanCalibration::RotationVector2Matrix(const Eigen::Vector3d& v)
{
	double s = std::sqrt(v.dot(v));
	Eigen::Vector3d axis = v / s;
	Eigen::AngleAxisd r(s, axis);
	return r.toRotationMatrix();
}

Eigen::Matrix3f LineScanCalibration::Matrixd2f(const Eigen::Matrix3d& matrix)
{
	Eigen::Matrix3f matrix3f;
	matrix3f << matrix(0, 0), matrix(0, 1), matrix(0, 2),
		matrix(1, 0), matrix(1, 1), matrix(1, 2),
		matrix(2, 0), matrix(2, 1), matrix(2, 2);
	return matrix3f;
}

double LineScanCalibration::ReprojectError(std::vector<cv::Point2d>& pt2D, std::vector<cv::Point3d>& pt3D, Eigen::Matrix3d& R,Eigen::Vector3d& T,Eigen::Vector3d& k, double& Fy, double& vc)
{
	if (pt2D.size() != pt3D.size()) {
		std::cerr << "2D Not Equal 3D..." << std::endl;
		exit(-1);
	}
	/*获取畸变参数*/
	const double k1 = k(0, 0);
	const double k2 = k(1, 0);
	const double k3 = k(2, 0);
	const int ptsNum = pt3D.size();
	/*构造外参变换矩阵*/
	Eigen::Matrix4d Tran = 
		Eigen::Matrix<double,4,4>::Identity();
	Tran.block(0, 0, 3, 3) = R;
	Tran.block(0, 3, 3, 1) = T;
	/*构造内参矩阵*/
	Eigen::Matrix<double, 3, 4> K = 
		Eigen::Matrix<double,3,4>::Zero();
	K(0, 0) = K(2, 2) = 1.;
	K(1, 1) = Fy;
	K(1, 2) = vc;
	/*构造M矩阵*/
	Eigen::Matrix<double, 3, 4> M = K*Tran;
	/*计算重投影误差*/
	float error = 0.;
	for (int it = 0; it < ptsNum; ++it) { 
		double u = pt2D[it].x;
		double v = pt2D[it].y;
		double v_ = (v - vc) / Fy;
		v_ -= k1*v_*v_*v_ + k2*v_*v_*v_*v_*v_ + k3*v_*v_;
		v = v_*Fy + vc;
		Eigen::Vector4d Pw;
		Pw << pt3D[it].x, pt3D[it].y, pt3D[it].z, 1.;
		Eigen::Vector3d reproPw = M*Pw;
		const double u_repro = reproPw(0) / reproPw(2);
		const double v_repro = reproPw(1) / reproPw(2);
		error += (std::sqrt)((std::powf)(u_repro-u, 2) +
			(std::powf)(v_repro- v, 2));
	}
	return error / float(ptsNum);
}

double LineScanCalibration::ReprojectError(std::vector<cv::Point2d>& pt2D, std::vector<cv::Point3d>& pt3D, Eigen::Matrix<double,2,4>& M)
{
	if (pt2D.size() != pt3D.size()) {
		std::cerr << "2D Not Equal 3D..." << std::endl;
		exit(-1);
	}
	const int ptsNum = pt3D.size();
	/*获取参数*/
	const double m21 = M(0, 0);
	const double m22 = M(0, 1);
	const double m23 = M(0, 2);
	const double m24 = M(0, 3);
	const double m31 = M(1, 0);
	const double m32 = M(1, 1);
	const double m33 = M(1, 2);
	const double m34 = M(1, 3);
	/*计算重投影误差*/
	float error = 0.;
	for (int it = 0; it < ptsNum; ++it) {
		const double Xw = pt3D[it].x;
		const double Yw = pt3D[it].y;
		const double Zw = pt3D[it].z;
		const double v = pt2D[it].y;
		double v_ = (m21*Xw + m22*Yw + m23*Zw + m24) /
			(m31*Xw + m32*Yw + m33*Zw + m34);
		error += (std::abs)(v-v_);
	}
	return error / ptsNum;
}

void LineScanCalibration::SaveDebugImage()
{
	for (auto pt : mObjectPoints[0]) {
		cv::circle(mImageDebug,cv::Point2f(pt.x,pt.y),4,cv::Scalar(0,0,0),1,8,0);
	}
	const int featuresNum = mObjectPoints[0].size();
	cv::Point2f pt1(mObjectPoints[0][0].x, mObjectPoints[0][0].y);
	cv::Point2f pt2(mObjectPoints[0][featuresNum - 1].x, mObjectPoints[0][featuresNum - 1].y);
	cv::line(mImageDebug, pt1, pt2, cv::Scalar(0, 0, 0), 2, 8, 0);
	std::string saveFile = mDebugPath + "\\debug.bmp";
	cv::imwrite(saveFile, mImageDebug);
}

void LineScanCalibration::SaveGroundTruth()
{
	/*获取相机坐标参数*/
	mGroudTruth.clear();
	const int imagesNum = mObjectPoints.size();
	const int featuresNum = mObjectPoints.begin()->size();
	for (int iti = 0; iti < imagesNum; ++iti) {
		for (int itj = 0; itj < featuresNum; ++itj) {
			mGroudTruth.push_back(mObjectPoints[iti][itj]);
		}
	}
}

void LineScanCalibration::SaveWorldPointsBeforeOptimized(LineScanPara& cameraPara)
{
	/*获取相机标定参数*/
	Eigen::Matrix3d R = cameraPara.R;
	Eigen::Vector3d T = cameraPara.T;
	const double k1 = cameraPara.k1;
	const double k2 = cameraPara.k2;
	const double k3 = cameraPara.k3;
	const double fy = cameraPara.Fy;
	const double vc = cameraPara.vc;
	/*获取畸变参数矩阵*/
	Eigen::Vector3d dis;
	dis << k1, k2, k3;
	/*构造内参矩阵*/
	Eigen::Matrix<double, 3, 3> K =
		Eigen::Matrix<double, 3, 3>::Zero();
	K(0, 0) = K(2, 2) = 1.;
	K(1, 1) = fy;
	K(1, 2) = vc;
	/*计算3D坐标*/
	mWorldPointsBeforeOptimized.clear();
	const int imagesNum = mImagePoints.size();
	const int featuresNum = mImagePoints.begin()->size();
	for (int iti = 0; iti < imagesNum; ++iti) {
		for (int itj = 0; itj < featuresNum; ++itj) {
			/*获取特征像素坐标值*/
			cv::Point3d x3D;
			double scale = CalculateScale(mImagePoints[iti][itj], R, T, K, dis, x3D, mObjectPoints[iti][itj].z);
			mWorldPointsBeforeOptimized.push_back(x3D);
			//std::cout << scale << std::endl;
		}
	}
}

void LineScanCalibration::SaveWorldPointsAfterOptimized(LineScanPara& cameraPara)
{
	/*获取相机标定参数*/
	Eigen::Matrix3d R = cameraPara.R;
	Eigen::Vector3d T = cameraPara.T;
	const double k1 = cameraPara.k1;
	const double k2 = cameraPara.k2;
	const double k3 = cameraPara.k3;
	const double fy = cameraPara.Fy;
	const double vc = cameraPara.vc;
	/*获取畸变参数矩阵*/
	Eigen::Vector3d dis;
	dis << k1, k2, k3;
	/*构造内参矩阵*/
	Eigen::Matrix<double, 3, 3> K =
		Eigen::Matrix<double, 3, 3>::Zero();
	K(0, 0) = K(2, 2) = 1.;
	K(1, 1) = fy;
	K(1, 2) = vc;
	/*计算3D坐标*/
	mWorldPointsAfterOptimized.clear();
	const int imagesNum = mImagePoints.size();
	const int featuresNum = mImagePoints.begin()->size();
	for (int iti = 0; iti < imagesNum; ++iti) {
		for (int itj = 0; itj < featuresNum; ++itj) {
			/*获取特征像素坐标值*/
			cv::Point3d x3D;
			double scale = CalculateScale(mImagePoints[iti][itj], R, T, K, dis, x3D, mObjectPoints[iti][itj].z);
			mWorldPointsAfterOptimized.push_back(x3D);
			//std::cout << scale << std::endl;
		}
	}
}

double LineScanCalibration::CalculateScale(cv::Point2d& imagePoint,
	 Eigen::Matrix3d& R,
	 Eigen::Vector3d& t,
	 Eigen::Matrix3d& K,
	 Eigen::Vector3d& dis,
	 cv::Point3d& x3D,
	 double Zw)
{
	/*去除相机像素坐标畸变*/
	Eigen::Vector2d unImagePoint;
	Eigen::Vector2d eigenPoint;
	eigenPoint << imagePoint.x, imagePoint.y;
	UndistortionKeys(eigenPoint, unImagePoint, K, dis);
	/*计算相机尺度*/
	Eigen::Vector3d leftSideMatrix = Eigen::Vector3d::Zero();
	Eigen::Vector3d rightSideMatrix = Eigen::Vector3d::Zero();
	Eigen::Vector3d imageHomo = Eigen::Vector3d::Ones();
	imageHomo(0) = unImagePoint(0);
	imageHomo(1) = unImagePoint(1);
	leftSideMatrix = R.inverse() * K.inverse() * imageHomo;
	rightSideMatrix = R.inverse() * t;
	double scale = (Zw + rightSideMatrix(2, 0)) / leftSideMatrix(2, 0);
	/*计算相机坐标系下特征点3D坐标*/
	Eigen::Vector3d point3D = scale*R.inverse() * K.inverse() * imageHomo - R.inverse()*t;
	x3D = cv::Point3d(point3D(0), point3D(1), point3D(2));
	return scale;
}

void LineScanCalibration::UndistortionKeys(Eigen::Vector2d& vUnKeys, 
	Eigen::Vector2d& vKeys, 
	Eigen::Matrix3d& K, 
	Eigen::Vector3d& dis)
{
	/*获取相机内参&畸变参数*/
	const double fx = K(0, 0);
	const double fy = K(1, 1);
	const double cx = K(0, 2);
	const double cy = K(1, 2);
	const double k1 = dis(0, 0);
	const double k2 = dis(1, 0);
	const double k3 = dis(2, 0);
	/*获取相机像素坐标*/
	double u = vUnKeys(0);
	double v = vUnKeys(1);
	/*获取相机归一化相机系坐标*/
	double xp = (u - cx) / fx;
	double yp = (v - cy) / fy;
	/*获取像素坐标径向长度*/
	double r_2 = xp * xp + yp * yp;
	/*获取相机去畸变归一化坐标*/
	double xdis = xp;
	double ydis = yp - (k1*yp*yp*yp + k2*yp*yp*yp*yp*yp + k3*yp*yp);
	/*获取相机去畸变像素坐标*/
	double u_un = fx * xdis + cx;
	double v_un = fy * ydis + cy;
	vKeys = Eigen::Vector2d(u_un, v_un);
	return;
}
