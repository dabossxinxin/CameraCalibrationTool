#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

namespace CommonStruct
{
	struct LineFunction2D {
		double a;
		double b;
		double c;
		LineFunction2D() {
			a = 0.;
			b = 0.;
			c = 0.;
		}
	};
}

namespace CommonFunctions
{
	double norm(const cv::Mat& inputArray);

	cv::Mat normalize(const cv::Mat& inputArray);

	void ConditionPrint(const std::string&);

	CommonStruct::LineFunction2D ComputeLineFunction2D(const cv::Point2d& p1,const cv::Point2d& p2);
	
	/*使用一组2D点拟合直线方程，并评估直线方程的误差*/
	CommonStruct::LineFunction2D ComputeLineFunction2D(const std::vector<cv::Point2d>& pts);

	/*计算点到直线的距离*/
	double ComputeDistanceFrom2DL2P(const CommonStruct::LineFunction2D&, const cv::Point2d&);

	/*计算两条2D直线的交点*/
	cv::Point2d ComputeIntersectionPt(const CommonStruct::LineFunction2D&, const CommonStruct::LineFunction2D&);
	
	/*通过一条直线与直线上一点重新计算直线*/
	CommonStruct::LineFunction2D ComputeLineFunction2D(const CommonStruct::LineFunction2D&, const cv::Point2d&);

	/*计算数组的平均值*/
	template <class T>
	double Mean(std::vector<T>& vals)
	{
		const size_t size = vals.size();
		if (size == 0) return 0.;
		double meanVal = 0.;
		for (size_t it = 0; it < size; ++it) {
			meanVal += vals[it];
		}
		return meanVal /= size;
	}

	/*灰度质心法*/
	cv::Point2d GrayScaleCentroid(const unsigned char* const, const std::vector<int>&, const int, const int);

	/*计算两点之间的距离*/
	double ComputeDistanceP2P(const cv::Point2d&, const cv::Point2d&);
	double ComputeDistanceP2P(const cv::Point3d&, const cv::Point3d&);

	/*计算数组的平均值*/
	double Average(const std::vector<double>&);
	/*将PointXd转化为VectorXd*/
	void Point2d2Vector2d(const cv::Point2d& cvPt, Eigen::Vector2d& eigenPt);
	void Point3d2Vector3d(const cv::Point3d& cvPt, Eigen::Vector3d& eigenPt);
	/*绕某固定点旋转固定角度(逆时针)*/
	void RotatePoint(const cv::Point2d&, const cv::Point2d&, const float, cv::Point2d&);
	/*交换X和Y分量的值*/
	void ExchageXY(std::vector<std::vector<cv::Point2d>>&);
}

