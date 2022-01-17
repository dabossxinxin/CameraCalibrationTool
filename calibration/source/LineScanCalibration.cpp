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

void FeaturesPointExtract::DiagonalLine3DPoints()
{
	if (mFeatures2D.size() != mFeaturesNum) exit(-1);
	if (mFeatures3D.size() != mFeaturesNum) exit(-1);

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
	this->DiagonalLine3DPoints();
	
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
	//输出调试信息
	CommonFunctions::ConditionPrint("Start Initial Estimate...");	
	//函数运行前必进行的检查工作
	if (mObjectPoints.size() != mImagePoints.size()){
		std::cerr << "Object Points's Are Not Equal To Image Points's" << std::endl;
		return false;
	}
	if (mObjectPoints.empty() || mImagePoints.empty()) {
		std::cerr << "Features Points Is Empty" << std::endl;
		return false;
	}
	//根据2D点与3D点的坐标，估计投影矩阵M
	Eigen::MatrixXd sumPts(3,1); 
	sumPts(0, 0) = sumPts(1, 0) = sumPts(2, 0) = 0.;
	const int imagesNum = mObjectPoints.size();
	const int featuresNum = mObjectPoints.begin()->size();
	Eigen::MatrixXd C3(featuresNum*imagesNum, 3);
	Eigen::MatrixXd C5(featuresNum*imagesNum, 5);
	for (int i = 0; i < imagesNum; ++i) {
		for (int j = 0; j < featuresNum; ++j) {
			C3(i*featuresNum + j, 0) = double(-mImagePoints[i][j].y*mObjectPoints[i][j].x);
			C3(i*featuresNum + j, 1) = double(-mImagePoints[i][j].y*mObjectPoints[i][j].y);
			C3(i*featuresNum + j, 2) = double(-mImagePoints[i][j].y*mObjectPoints[i][j].z);
			
			C5(i*featuresNum + j, 0) = double(mObjectPoints[i][j].x);
			C5(i*featuresNum + j, 1) = double(mObjectPoints[i][j].y);
			C5(i*featuresNum + j, 2) = double(mObjectPoints[i][j].z);
			C5(i*featuresNum + j, 3) = 1.0;
			C5(i*featuresNum + j, 2) = double(mImagePoints[i][j].y);
			
			sumPts(0, 0) = sumPts(0 ,0) + double(mObjectPoints[i][j].x);
			sumPts(1, 0) = sumPts(1, 0) + double(mObjectPoints[i][j].y);
			sumPts(2, 0) = sumPts(2, 0) + double(mObjectPoints[i][j].z);
		}
	}
	sumPts /= featuresNum*imagesNum;
	Eigen::MatrixXd C3T = C3.transpose();
	Eigen::MatrixXd C5T = C5.transpose();
	Eigen::MatrixXd D = -C3T*C3 + C3T*C5*(C5T*C5).inverse()*C5T*C3;
	/*最小特征值对应的特征向量*/
	Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(D);
	Eigen::MatrixXcd evecs = es.eigenvectors();
	Eigen::MatrixXcd evals = es.eigenvalues();
	Eigen::MatrixXd evalsReal = evals.real();
	Eigen::MatrixXf::Index evalsMin;
	evalsReal.rowwise().sum().minCoeff(&evalsMin);
	Eigen::MatrixXd phi3 = Eigen::MatrixXd(3, 1);
	phi3(0, 0) = evecs.real()(0, evalsMin);
	phi3(1, 0) = evecs.real()(1, evalsMin);
	phi3(2, 0) = evecs.real()(2, evalsMin);
	Eigen::MatrixXd phi5 = -(C5T*C5).inverse()*C5T*C3*phi3;
	/*特征值组成投影矩阵*/
	Eigen::MatrixXd m3 = phi3;
	float m24 = phi5(3,0), m34 = phi5(4,0);
	Eigen::MatrixXd m2 = phi5.block(0, 0, 1, 3);

	//根据投影矩阵M，估计出初始的线扫相机内参外参
	Eigen::MatrixXd vc = m2.transpose()*m3;
	mCameraPara.vc = vc(0, 0);
	Eigen::MatrixXd Fy = m2.transpose()*skew(m3);
	mCameraPara.Fy = Fy.norm();
	Eigen::MatrixXd r2 = (m2-mCameraPara.vc*m3)/mCameraPara.Fy;
	Eigen::MatrixXd r3 = m3;
	Eigen::MatrixXd r1 = (r2.transpose()*skew(r3)).transpose();
	mCameraPara.T(1,0) = (m24-mCameraPara.vc*m34)/mCameraPara.Fy;
	mCameraPara.T(2,0) = m34;
	Eigen::MatrixXd t1 = r1.transpose()*sumPts;
	mCameraPara.T(0,0) = t1(0, 0);
	if (mCameraPara.T(2, 0) < 0) {
		std::cerr << "Initialize Error..." << std::endl;
		return false;
	}
	//输出调试信息
	CommonFunctions::ConditionPrint("End Initial Estimate");
	return true;
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
	const int imagesNum = mObjectPoints.size();
	const int featuresNum = mObjectPoints.begin()->size();
	Eigen::Matrix3f R = mCameraPara.R;
	Eigen::Vector3f rf = this->RotationMatrix2Vector(R);
	Eigen::Vector3f T = mCameraPara.T;
	Eigen::Vector3d r; r << rf(0), rf(1), rf(2);
	Eigen::Vector3d t; t << T(0), T(1), T(2);
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
	std::cout << "开始非线性优化..." << std::endl;
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = false;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
	options.preconditioner_type = ceres::JACOBI;
	options.max_num_iterations = 1000;
	options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
	options.function_tolerance = 1e-16;
	options.gradient_tolerance = 1e-10;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	if (!summary.IsSolutionUsable()){
		std::cerr << "非线性优化失败..." << std::endl;
	}
	else {
		/*打印非线性优化的相关信息*/
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
		/*确定线扫相机标定参数*/
		mCameraPara.Fy = k[0];
		mCameraPara.vc = k[1];
		mCameraPara.k1 = k[2];
		mCameraPara.k2 = k[3];
		mCameraPara.k3 = k[4];
		//mCameraPara.R = this->Matrixd2f(this->RotationVector2Matrix(r));
		//mCameraPara.T << t(0), t(1), t(3);
	}
	CommonFunctions::ConditionPrint("End Optimize Estimate");
	return true;
}

bool LineScanCalibration::Resolution()
{
	const float k1 = mCameraPara.k1;
	const float k2 = mCameraPara.k2;
	const float k3 = mCameraPara.k3;
	const float fx = 1.;
	const float fy = mCameraPara.Fy;
	const float cx = 0.;
	const float cy = mCameraPara.vc;
	const int imageSize = mImagePoints.size();
	const int featuresSize = mImagePoints.begin()->size();
	for (int i = 0; i < imageSize; ++i) {
		for (int j = 0; j < featuresSize; ++j) {
			
		}
	}
}

Eigen::MatrixXd LineScanCalibration::skew(Eigen::MatrixXd& vec)
{
	const double x = vec(0, 0);
	const double y = vec(1, 0);
	const double z = vec(2, 0);
	Eigen::MatrixXd res(3, 3);
	res(0, 0) = 0; res(0, 1) = -z; res(0, 2) = y;
	res(1, 0) = z; res(1, 1) = 0; res(1, 2) = -x;
	res(2, 0) = -y; res(2, 1) = x; res(2, 2) = 0;
	return res;
}

Eigen::Vector3f LineScanCalibration::RotationMatrix2Vector(const Eigen::Matrix3f& R)
{
	Eigen::AngleAxisf r;
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