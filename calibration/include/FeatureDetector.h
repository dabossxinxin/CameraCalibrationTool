#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include "RegionDetect.h"
#include "PointEllipseFitting.h"
#include "CeresCostFun.h"
#include "ZernikeMoment.h"
#include "DualEllipseFitting.h"

using namespace std;
using namespace Eigen;
using namespace MtZernikeMoment;
using namespace MtRegionDetect;
using namespace MtEllipseFitting;
using namespace MtDualEllipseFitting;

namespace MtFeatureDetector
{
	class FeatureDetectorFailure
	{
	public:
		FeatureDetectorFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class FeatureDetector
	{
	public:
		FeatureDetector(const cv::Mat& image,const int& row, const int& col):
			mImage(image),mRow(row),mCol(col)
		{
			if (mImage.channels() == 3)
			{
				cv::cvtColor(mImage, mGrayImage, cv::COLOR_BGR2GRAY);
			}
			else if (mImage.channels() == 1)
			{
				mGrayImage = mImage;
			}
		}

		void compute(std::vector<cv::Point2d>& features);
	private:
		cv::Mat mImage;
		cv::Mat mGrayImage;
		cv::Mat mGradientX;
		cv::Mat mGradientY;
		enum GradientMethod {Sobel = 1, Roberts = 2, Prewitt = 3};

		// iamge parameter
		const int mRow;
		const int mCol;

		// method
		cv::Mat SobelOperator(const cv::Mat& image, const string& flag);
		cv::Mat RobertsOperator(const cv::Mat& image, const string& flag);
		cv::Mat PrewittOperator(const cv::Mat& image, const string& flag);
		cv::Mat SubPixelOperator(const cv::Mat& image, const string& flag);

		int Otsu(const cv::Mat& image);
		void GrayMoment(const pair<cv::Mat,cv::Point2d>& image, float& u, float& v);
		void SortFeatures(std::vector<cv::Point2d>& features,const cv::Point2d& p1, const cv::Point2d& p2, const int& row, const int& col);
		void minRectangleRatio(const int& threshold, const pair<cv::Mat,cv::Point2d>& image,double& area, double& ratioRec, double& ratioAll);
		double computeDistance2D(const cv::Point2d& p1, const cv::Point2d& p2);
		double computeDistance2D(const cv::Point2d& p, const double& a, const double& b);
		double CheckRoundness(const cv::Mat& image);
		void calculateRectangleArea(const std::vector<cv::Point2d>& minRectangle, double& area);
		void CalculateGradient(const cv::Mat& image, const GradientMethod& method)throw(FeatureDetectorFailure);

		void GetClosetTwoPoint(const std::vector<cv::Point2d>& features, cv::Point2d& p1, cv::Point2d& p2);
		void CheckOrder(const std::vector<cv::Point2d>& features, std::vector<cv::Point2d>& sortedFeatures);
		void CheckDirection(std::vector<std::vector<pair<cv::Point2d,cv::Point2d>>>& features, const cv::Point2d& p1, const cv::Point2d& p2);
		void RotateFeatures(const std::vector<cv::Point2d>& benchmark, const std::vector<cv::Point2d>& features, std::vector<cv::Point2d>& sortedFeatures)throw(FeatureDetectorFailure);
		void GetCornerByPoint(const std::vector<cv::Point2d>& features, const cv::Point2d& mark, cv::Point2d& feature);
		bool Normalize(Eigen::MatrixXd& P, Eigen::Matrix3d& T);
		void GetCentroid(const std::vector<cv::Point2d>& features, cv::Point2d& gravity);
		void SortWithPolarCoordinate(const std::vector<cv::Point2d>& features, std::vector<cv::Point2d>& sortedFeatures)throw(FeatureDetectorFailure);
		void SortWithXYAxis(const std::vector<pair<cv::Point2d,cv::Point2d>>& features,const cv::Point2d& p1,const cv::Point2d& p2,
							const int& row, const int& col, std::vector<cv::Point2d>& sortedFeatures)throw(FeatureDetectorFailure);
		void FindHomography(const std::vector<cv::Point2d>& features, Eigen::Matrix3d& homography);
		Eigen::VectorXd solveHomographyDLT(const Eigen::MatrixXd& srcPoints, const Eigen::MatrixXd& dstPoints)throw(FeatureDetectorFailure);
		
	};

	class MaxHeap
	{
	public:
		
		MaxHeap()
		{
			heap.resize(2);
			heapSize = 0;
		}

		void push(const pair<int, double>& x)
		{
			if (heapSize == 0) heap.resize(2);
			if (heapSize == heap.size() - 1) changeLength();

			int currentNode = ++heapSize;
			while (currentNode != 1 && x.second > heap[currentNode / 2].second)
			{
				heap[currentNode] = heap[currentNode / 2];
				currentNode /= 2;
			}
			heap[currentNode] = x;
		}

		void pop()
		{
			int deleteIndex = heapSize;
			pair<int,double> lastElement = heap[heapSize--];
			
			int currentNode = 1;
			int chirld = 2;
			while (chirld <= heapSize)
			{
				if (chirld < heapSize && heap[chirld].second < heap[chirld + 1].second) chirld++;
				if (lastElement.second >= heap[chirld].second) break;
				
				heap[currentNode] = heap[chirld];
				currentNode = chirld;
				chirld *= 2;
			}
			heap[currentNode] = lastElement;
			heap.erase(heap.begin() + deleteIndex);
		}

		int size()
		{
			return heapSize;
		}

		pair<int,double> top()
		{
			if (!this->empty()) return heap[1];
			else exit(-1);
		}

		bool empty()
		{
			if (heapSize == 0) return true;
			return false;
		}
		
	private: 
		std::vector<pair<int, double>> heap;   
		int heapSize;

		void changeLength()
		{
			std::vector<pair<int, double>> heapTemp;
			heapTemp = heap;
			heap.resize(2 * heap.size());
			
			for (int i = 0; i < heapTemp.size(); i++)
			{
				heap[i] = heapTemp[i];
			}
		}
	};

	class MinHeap
	{
	public:

		MinHeap()
		{
			heap.resize(2);
			heapSize = 0;
		}

		void push(const pair<int, double>& x)
		{
			if (heapSize == 0) heap.resize(2);
			if (heapSize == heap.size() - 1) changeLength();

			int currentNode = ++heapSize;
			while (currentNode != 1 && x.second < heap[currentNode / 2].second)
			{
				heap[currentNode] = heap[currentNode / 2];
				currentNode /= 2;
			}
			heap[currentNode] = x;
		}

		void pop()
		{
			int deleteIndex = heapSize;
			pair<int, double> lastElement = heap[heapSize--];

			int currentNode = 1;
			int chirld = 2;
			while (chirld <= heapSize)
			{
				if (chirld < heapSize && heap[chirld].second > heap[chirld + 1].second) chirld++;
				if (lastElement.second <= heap[chirld].second) break;

				heap[currentNode] = heap[chirld];
				currentNode = chirld;
				chirld *= 2;
			}
			heap[currentNode] = lastElement;
			heap.erase(heap.begin() + deleteIndex);
		}

		int size()
		{
			return heapSize;
		}

		pair<int, double> top()
		{
			if (!this->empty()) return heap[1];
			else exit(-1);
		}

		bool empty()
		{
			if (heapSize == 0) return true;
			return false;
		}

	private:
		std::vector<pair<int, double>> heap;
		int heapSize;

		void changeLength()
		{
			std::vector<pair<int, double>> heapTemp;
			heapTemp = heap;
			heap.resize(2 * heap.size());

			for (int i = 0; i < heapTemp.size(); i++)
			{
				heap[i] = heapTemp[i];
			}
		}
	};
}