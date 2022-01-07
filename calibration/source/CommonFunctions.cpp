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
}