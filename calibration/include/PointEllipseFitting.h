#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

#define PI 3.1415926535

namespace MtEllipseFitting
{
	class EllipseSolverFailure
	{
	public:
		EllipseSolverFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class EllipseSolver
	{
	public:
		EllipseSolver(const std::vector<float>& x, const std::vector<float>& y, const bool& gravity):
			mX(x),mY(y),mbGravity(gravity){}
		void compute()throw(EllipseSolverFailure);
		void getEllipsePara(cv::Mat& para) { para = mEllipsePara; }
	private:
		const std::vector<float> mX;
		const std::vector<float> mY;
		const bool mbGravity;
		double mGravityX;
		double mGravityY;
		cv::Mat1d mEllipsePara;
		

		// method
		void Gravity(const std::vector<float>& input, std::vector<float>& output, const string& direction);
		void GetSmallestEigenVal(const cv::Mat1d& S, int& index);
		cv::Mat1d convertConicToParametric(cv::Mat1d& par);
		int sign(const double& val) { return (val > 0) - (val < 0); }
		void constructD(cv::Mat1d& D1, cv::Mat1d& D2, const bool& gravity)throw(EllipseSolverFailure);
	};

	class DataTran
	{
	public:
	private:
	};
}