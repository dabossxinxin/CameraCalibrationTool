#pragma once

#include <iostream>
#include <opencv2/core.hpp>

namespace CommonStruct
{
	struct LineFunction2D {
		float a;
		float b;
		float c;
		LineFunction2D() {
			a = 0.;
			b = 0.;
			c = 0.;
		}
	};
}

namespace CommonFunctions
{
	float norm(const cv::Mat& inputArray);

	cv::Mat normalize(const cv::Mat& inputArray);

	void ConditionPrint(const std::string&);

	CommonStruct::LineFunction2D& ComputeLineFunction2D(const cv::Point2f& p1,const cv::Point2f& p2);
}

