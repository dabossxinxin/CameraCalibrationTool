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

	}
}