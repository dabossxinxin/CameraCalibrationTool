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
	
	/*ʹ��һ��2D�����ֱ�߷��̣�������ֱ�߷��̵����*/
	CommonStruct::LineFunction2D ComputeLineFunction2D(const std::vector<cv::Point2d>& pts);

	/*����㵽ֱ�ߵľ���*/
	double ComputeDistanceFrom2DL2P(const CommonStruct::LineFunction2D&, const cv::Point2d&);

	/*��������2Dֱ�ߵĽ���*/
	cv::Point2d ComputeIntersectionPt(const CommonStruct::LineFunction2D&, const CommonStruct::LineFunction2D&);
	
	/*ͨ��һ��ֱ����ֱ����һ�����¼���ֱ��*/
	CommonStruct::LineFunction2D ComputeLineFunction2D(const CommonStruct::LineFunction2D&, const cv::Point2d&);

	/*���������ƽ��ֵ*/
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

	/*�Ҷ����ķ�*/
	cv::Point2d GrayScaleCentroid(const unsigned char* const, const std::vector<int>&, const int, const int);

	/*��������֮��ľ���*/
	double ComputeDistanceP2P(const cv::Point2d&, const cv::Point2d&);
	double ComputeDistanceP2P(const cv::Point3d&, const cv::Point3d&);

	/*���������ƽ��ֵ*/
	double Average(const std::vector<double>&);
	/*��PointXdת��ΪVectorXd*/
	void Point2d2Vector2d(const cv::Point2d& cvPt, Eigen::Vector2d& eigenPt);
	void Point3d2Vector3d(const cv::Point3d& cvPt, Eigen::Vector3d& eigenPt);
	/*��ĳ�̶�����ת�̶��Ƕ�(��ʱ��)*/
	void RotatePoint(const cv::Point2d&, const cv::Point2d&, const float, cv::Point2d&);
	/*����X��Y������ֵ*/
	void ExchageXY(std::vector<std::vector<cv::Point2d>>&);
}

