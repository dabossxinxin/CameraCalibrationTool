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
	
	/*ʹ��һ��2D�����ֱ�߷��̣�������ֱ�߷��̵����*/
	CommonStruct::LineFunction2D& ComputeLineFunction2D(const std::vector<cv::Point2f>& pts);

	/*����㵽ֱ�ߵľ���*/
	float ComputeDistanceFrom2DL2P(const CommonStruct::LineFunction2D&, const cv::Point2f&);

	/*��������2Dֱ�ߵĽ���*/
	cv::Point2f& ComputeIntersectionPt(const CommonStruct::LineFunction2D&, const CommonStruct::LineFunction2D&);

	/*���������ƽ��ֵ*/
	template <class T>
	float Mean(std::vector<T>&);

	/*�Ҷ����ķ�*/
	cv::Point2f& GrayScaleCentroid(const unsigned char* const, const std::vector<int>&, const int, const int);
}

