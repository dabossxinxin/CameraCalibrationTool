#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

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
	
	/*使用一组2D点拟合直线方程，并评估直线方程的误差*/
	CommonStruct::LineFunction2D& ComputeLineFunction2D(const std::vector<cv::Point2f>& pts);

	/*计算点到直线的距离*/
	float ComputeDistanceFrom2DL2P(const CommonStruct::LineFunction2D&, const cv::Point2f&);

	/*计算两条2D直线的交点*/
	cv::Point2f& ComputeIntersectionPt(const CommonStruct::LineFunction2D&, const CommonStruct::LineFunction2D&);

	/*计算数组的平均值*/
	template <class T>
	float Mean(std::vector<T>&);

	/*灰度质心法*/
	cv::Point2f& GrayScaleCentroid(const unsigned char* const, const std::vector<int>&, const int, const int);
}

