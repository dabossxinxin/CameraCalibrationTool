#pragma once

#include <iostream>
#include <opencv2/core.hpp>

namespace CommonFunctions
{
	float norm(const cv::Mat& inputArray);

	cv::Mat normalize(const cv::Mat& inputArray);

	void ConditionPrint(const std::string&);
}