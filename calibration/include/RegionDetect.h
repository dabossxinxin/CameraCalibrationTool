#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;

#define EdgePixel 16

namespace MtRegionDetect
{
	struct Pixel
	{
		int u;
		int v;
		int value;

		Pixel(const int& u, const int& v, const int& value) :u(u), v(v), value(value) {};
	};

	typedef std::vector<cv::Mat> MatrixMat;
	typedef std::vector<pair<cv::Mat, cv::Point2f>> MatrixP;

	class RegionDetectFailure
	{
	public:
		RegionDetectFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class RegionDetect
	{
	public:
		void setInputImage(const cv::Mat& image) { mImage = image; }
		void compute(MatrixMat& region, const int& threshold)throw(RegionDetectFailure);
		void compute(MatrixP& region, const int& threshold)throw(RegionDetectFailure);
		
	private:
		cv::Mat mImage;
		cv::Mat mGrayImage;

		enum BFSMethod {Iteration = 1, Recursion = 2};

		vector<int> dx;
		vector<int> dy;
		vector<vector<bool>> mIsVisited;

		// method
		void Initialize();
		void InitialImage(cv::Mat& image, const int& value);
		int Otsu(const cv::Mat& image);
		void BFS_Iteration(const cv::Mat& input, const int& th, int& count, std::vector<cv::Mat>& output)throw(RegionDetectFailure);
		void BFS_Iteration(const cv::Mat& input, const int& th, int& count, std::vector<std::vector<pair<cv::Point2i, int>>>& output)throw(RegionDetectFailure);
		void BFS_Recursion(const cv::Mat& input, const int& row, const int& col, const int& th, cv::Mat& output);
		int RegionGrowth(const cv::Mat& image, const int& th, MatrixMat& region, const BFSMethod& method)throw(RegionDetectFailure);
		int RegionGrowth(const cv::Mat& image, const int& th, MatrixP& region)throw(RegionDetectFailure);
	};
}