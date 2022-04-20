#include "CommonFunctions.h"

#define PRINT_INFO

namespace CommonFunctions
{
	double norm(const cv::Mat& inputArray)
	{
		double normVal = 0.;
		const int rows = inputArray.rows;
		for (int it = 0; it < rows; ++it)
			normVal += inputArray.at<double>(it, 0)*inputArray.at<double>(it, 0);
		return sqrt(normVal);
	}

	cv::Mat normalize(const cv::Mat& inputArray)
	{
		const int rows = inputArray.rows;
		double normVal = norm(inputArray);
		cv::Mat normalized = cv::Mat(rows, 1, CV_64FC1, cv::Scalar(0.));
		for (int it = 0; it < rows; ++it)
			normalized.at<double>(it,0) = inputArray.at<double>(it,0) / normVal;
		return normalized;
	}

	void ConditionPrint(const std::string& str)
	{
#ifdef PRINT_INFO
		std::cout << str << std::endl;
#endif
	}

	CommonStruct::LineFunction2D ComputeLineFunction2D(const cv::Point2d& p1, const cv::Point2d& p2)
	{
		CommonStruct::LineFunction2D line;
		const double x1 = p1.x;
		const double y1 = p1.y;
		const double x2 = p2.x;
		const double y2 = p2.y;
		line.a = y2 - y1;
		line.b = x1 - x2;
		line.c = x2*y1 - x1*y2;
		return line;
	}

	CommonStruct::LineFunction2D ComputeLineFunction2D(const CommonStruct::LineFunction2D& line, const cv::Point2d& pt)
	{
		const double a = line.a;
		const double b = line.b;
		const double x = pt.x;
		const double y = pt.y;
		CommonStruct::LineFunction2D lineOut;
		lineOut.a = a;
		lineOut.b = b;
		lineOut.c = -(a*x + b*y);
		return lineOut;
	}

	CommonStruct::LineFunction2D ComputeLineFunction2D(const std::vector<cv::Point2d>& pts)
	{
		const int Num = pts.size();
		Eigen::MatrixXd A = Eigen::MatrixXd(Num, 3);
		for (size_t it = 0; it < Num; ++it) {
			A(it, 0) = pts[it].x;
			A(it, 1) = pts[it].y;
			A(it, 2) = 1.;
		}
		Eigen::MatrixXd N = A.transpose()*A;
		Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> es(N);
		Eigen::MatrixXcd evecs = es.eigenvectors();
		Eigen::MatrixXcd evals = es.eigenvalues();
		Eigen::MatrixXd evalsReal = evals.real();
		Eigen::MatrixXf::Index evalsMin;
		evalsReal.rowwise().sum().minCoeff(&evalsMin);
		CommonStruct::LineFunction2D line;
		//得到最小特征值对应的特征向量
		line.a = evecs.real()(0, evalsMin);
		line.b = evecs.real()(1, evalsMin);
		line.c = evecs.real()(2, evalsMin);
		return line;
	}

	double ComputeDistanceFrom2DL2P(const CommonStruct::LineFunction2D& line, const cv::Point2d& pt)
	{
		const double a = line.a;
		const double b = line.b;
		const double c = line.c;
		const double x = pt.x;
		const double y = pt.y;
		return ((std::abs)(a*x+b*y+c))/(std::sqrt(a*a+b*b));
	}

	cv::Point2d ComputeIntersectionPt(const CommonStruct::LineFunction2D& l1, const CommonStruct::LineFunction2D& l2)
	{
		const double a1 = l1.a;
		const double b1 = l1.b;
		const double c1 = l1.c;
		const double a2 = l2.a;
		const double b2 = l2.b;
		const double c2 = l2.c;
		if (a1*b2 == a2*b1) {
			std::cerr << "两条直线平行，无法求解交点..." << std::endl;
			exit(-1);
		}

		Eigen::Matrix<double, 2, 2> N;
		N(0, 0) = a1;N(0, 1) = b1;
		N(1, 0) = a2;N(1, 1) = b2;
		Eigen::Matrix<double, 2, 1> C;
		C(0, 0) = -c1;C(1, 0) = -c2;
		Eigen::Matrix<double, 2, 1> result = N.inverse()*C;
		return cv::Point2d(result(0, 0), result(1, 0));
	}

	cv::Point2d GrayScaleCentroid(const unsigned char* const pImage, const std::vector<int>& roi, const int rows, const int cols)
	{
		if (pImage == nullptr) exit(-1);
		if (roi.empty()) exit(-1);
		const int size = roi.size();
		double m10 = 0., m01 = 0., m00 = 0.;
		for (size_t it = 0; it < size; ++it) {
			const int imageRow = roi[it]/cols;
			const int imageCol = roi[it]%cols;
			m10 += imageCol*pImage[it];//x
			m01 += imageRow*pImage[it];//y
			m00 += pImage[it];
		}
		if (m00 == 0.) exit(-1);
		return cv::Point2d(m10/m00, m01/m00);
	}

	double ComputeDistanceP2P(const cv::Point2d& p1, const cv::Point2d& p2)
	{
		const double dx = p1.x - p2.x;
		const double dy = p1.y - p2.y;
		return std::sqrt(dx*dx + dy*dy);
	}

	double ComputeDistanceP2P(const cv::Point3d& p1, const cv::Point3d& p2)
	{
		const double dx = p1.x - p2.x;
		const double dy = p1.y - p2.y;
		const double dz = p1.z - p2.z;
		return std::sqrt(dx*dx + dy*dy + dz*dz);
	}

	double Average(const std::vector<double>& vec)
	{
		if (vec.empty()) return -1;
		double average = 0.;
		const int size = vec.size();
		for (int it = 0; it < size; ++it) {
			average += vec[it];
		}
		average /= size;
		return average;
	}

	void Point2d2Vector2d(const cv::Point2d& cvPt, Eigen::Vector2d& eigenPt)
	{
		eigenPt(0, 0) = cvPt.x;
		eigenPt(1, 0) = cvPt.y;
		return;
	}

	void Point3d2Vector3d(const cv::Point3d& cvPt, Eigen::Vector3d& eigenPt)
	{
		eigenPt(0, 0) = cvPt.x;
		eigenPt(1, 0) = cvPt.y;
		eigenPt(2, 0) = cvPt.z;
		return;
	}

	void RotatePoint(const cv::Point2d& inPt, const cv::Point2d& ori, const float theta, cv::Point2d& outPt)
	{
		outPt.x = ori.x + (inPt.x - ori.x)*cos(theta) - (inPt.y - ori.y)*sin(theta);
		outPt.y = ori.y + (inPt.y - ori.y)*cos(theta) + (inPt.x - ori.x)*sin(theta);
		return;
	}

	void ExchageXY(std::vector<std::vector<cv::Point2d>>& pts)
	{
		for (int iti = 0; iti < pts.size(); ++iti) {
			for (int itj = 0; itj < pts[iti].size(); ++itj) {
				double tmp = pts[iti][itj].x;
				pts[iti][itj].x = pts[iti][itj].y;
				pts[iti][itj].y = tmp;
			}
		}
	}

	float ComputeDistanceP2P(const cv::Point2f& p1, const cv::Point2f& p2)
	{
		const float dx = p1.x - p2.x;
		const float dy = p1.y - p2.y;
		return std::sqrt(dx*dx+dy*dy);
	}

	float ComputeDistanceP2P(const cv::Point3f& p1, const cv::Point3f& p2)
	{
		const float dx = p1.x - p2.x;
		const float dy = p1.y - p2.y;
		const float dz = p1.z - p2.z;
		return std::sqrt(dx*dx+dy*dy+dz*dz);
	}

	float Average(const std::vector<float>& vec)
	{
		if (vec.empty()) return -1;
		float average = 0.;
		const int size = vec.size();
		for (int it = 0; it < size; ++it) {
			average += vec[it];
		}
		average /= size;
		return average;
	}
}