#include "CommonFunctions.h"

#define PRINT_INFO

namespace CommonFunctions
{
	float norm(const cv::Mat& inputArray)
	{
		float normVal = 0.;
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

	CommonStruct::LineFunction2D& ComputeLineFunction2D(const cv::Point2f& p1, const cv::Point2f& p2)
	{
		CommonStruct::LineFunction2D line;
		const float x1 = p1.x;
		const float y1 = p1.y;
		const float x2 = p2.x;
		const float y2 = p2.y;
		line.a = y2 - y1;
		line.b = x1 - x2;
		line.c = x2*y1 - x1*y2;
		return line;
	}

	CommonStruct::LineFunction2D& ComputeLineFunction2D(const std::vector<cv::Point2f>& pts)
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

	float ComputeDistanceFrom2DL2P(const CommonStruct::LineFunction2D& line, const cv::Point2f& pt)
	{
		const float a = line.a;
		const float b = line.b;
		const float c = line.c;
		const float x = pt.x;
		const float y = pt.y;
		return ((std::abs)(a*x+b*y+c))/(std::sqrt(a*a+b*b));
	}

	cv::Point2f& ComputeIntersectionPt(const CommonStruct::LineFunction2D& l1, const CommonStruct::LineFunction2D& l2)
	{
		const float a1 = l1.a;
		const float b1 = l1.b;
		const float c1 = l1.c;
		const float a2 = l2.a;
		const float b2 = l2.b;
		const float c2 = l2.c;
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
		return cv::Point2f(result(0, 0), result(1, 0));
	}

	template <class T>
	float Mean(std::vector<T>& vals)
	{
		const size_t size = vals.size();
		if (size == 0) return 0.;
		float meanVal = 0.;
		for (size_t it = 0; it < size; ++it) {
			meanVal += val[it];
		}
		return meanVal /= size;
	}

	cv::Point2f& GrayScaleCentroid(const unsigned char* const pImage, const std::vector<int>& roi, const int rows, const int cols)
	{
		if (pImage == nullptr) exit(-1);
		if (roi.empty()) exit(-1);
		const int size = roi.size();
		float m10 = 0., m01 = 0., m00 = 0.;
		for (size_t it = 0; it < size; ++it) {
			const int imageRow = roi[it]/cols;
			const int imageCol = roi[it]%cols;
			m10 += imageCol*pImage[it];//x
			m01 += imageRow*pImage[it];//y
			m00 += pImage[it];
		}
		if (m00 == 0.) exit(-1);
		return cv::Point2f(m10/m00, m01/m00);
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