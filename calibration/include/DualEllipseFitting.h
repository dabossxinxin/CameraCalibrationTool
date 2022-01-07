#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;

#define NO_ERROR 0
#define NOT_A_PROPER_ELLIPSE -1
#define NO_ELLIPSES_DETECTED -2
#define PI 3.1415926535

namespace MtDualEllipseFitting
{
	class EllipseSolver
	{
	public:
		EllipseSolver(const cv::Mat& roi):mROI(roi){}
		
		~EllipseSolver()
		{
			mROI.release();
			mPara.release();
		}

		void getConicPara(cv::Mat1d& para) { para = mPara; }

		/*
		 * description: This Function Fits the Ellipse to the contour points using Dual Conic method
		 * warning: tentative
		 * parameter: roi_pixel - pixel locations of the ellipse
		 * parameter: dx dy - x and y gradients in the image
		 * parameter: para - ellipse in the conic form [x y a b theta] 
		 * parameter: Normalization - when this flag is set, the input data is Normalized
		 */
		int compute(const bool& Normalization);
	private:
		// method
		cv::Mat1d convertConicToParametric(cv::Mat& par);
		int otsu(const cv::Mat& image);
		int sign(const double& val) { return (val > 0) - (val < 0); }
		cv::Mat Roberts(const cv::Mat& image, const string& flag);
		cv::Mat Sobel(const cv::Mat& image, const string& flag);
		cv::Mat Prewitt(const cv::Mat& image, const string& flag);

		/*
		 * description: This Function Fits the Ellipse to the contour points using Dual Conic method
		 * warning: tentative
		 * parameter: roi_pixel - pixel locations of the ellipse
		 * parameter: dx dy - x and y gradients in the image
		 * parameter: conic - ellipse in the conic form [A B C D E F]
		 * parameter: Normalization - when this flag is set, the input data is Normalized
		 */
		int DualConicFitting(const std::vector<cv::Point_<double>>& roi_pixel, const cv::Mat& dx, const cv::Mat& dy, cv::Mat& conic, const bool& Normalization);

		cv::Mat mROI;	// region of interest
		cv::Mat mGrayROI;
		cv::Mat1d mPara;	// parameter of ellipse
	};
}