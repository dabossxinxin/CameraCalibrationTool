#pragma once
#include "ZernikeMoment.h"

using namespace std;

namespace MtZernikeMoment
{
	ZernikeMomentFailure::ZernikeMomentFailure(const char* msg) :Message(msg) {};
	const char* ZernikeMomentFailure::GetMessage() { return Message; };

	void ZernikeMoment::compute(std::vector<float>& x, std::vector<float>& y)throw(ZernikeMomentFailure)
	{
		if (mImage.channels() == 3)
			cv::cvtColor(mImage,mGrayImage,cv::COLOR_RGB2GRAY);
		else
			mGrayImage = mImage.clone();
		
		x.clear(); y.clear();

		//cv::Mat medianImage1,medianImage2;
		//MedianBlur(mGrayImage, medianImage1, 5);
		//cv::medianBlur(mGrayImage, medianImage2, 5);

		cv::Mat medianImage2 = mGrayImage.clone();

		cv::Mat NewAdaThresImage = mGrayImage.clone();
		cv::Mat cannyImage = mGrayImage.clone();
		//cv::adaptiveThreshold(medianImage2, NewAdaThresImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 7, 4);
		cv::Canny(medianImage2, cannyImage, 3, 9, 3);
		std::vector<cv::Point2d> SubEdgePoints;

		cv::Mat ZerImageM00;
		//cv::filter2D(NewAdaThresImage, ZerImageM00, CV_64F, M00, cv::Point(-1, -1), 0,cv::BORDER_DEFAULT);
		cv::filter2D(cannyImage, ZerImageM00, CV_64F, M00, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

		cv::Mat ZerImageM11R;
		//cv::filter2D(NewAdaThresImage, ZerImageM11R, CV_64F, M11R, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D( cannyImage, ZerImageM11R, CV_64F, M11R, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat ZerImageM11I;
		//cv::filter2D(NewAdaThresImage, ZerImageM11I, CV_64F, M11I, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D( cannyImage, ZerImageM11I, CV_64F, M11I, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat ZerImageM20;
		//cv::filter2D(NewAdaThresImage, ZerImageM20, CV_64F, M20, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D( cannyImage, ZerImageM20, CV_64F, M20, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat ZerImageM31R;
		//cv::filter2D(NewAdaThresImage, ZerImageM31R, CV_64F, M31R, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(cannyImage, ZerImageM31R, CV_64F, M31R, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat ZerImageM31I;
		//cv::filter2D(NewAdaThresImage, ZerImageM31I, CV_64F, M31I, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(cannyImage, ZerImageM31I, CV_64F, M31I, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		cv::Mat ZerImageM40;
		//cv::filter2D(NewAdaThresImage, ZerImageM40, CV_64F, M40, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(cannyImage, ZerImageM40, CV_64F, M40, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

		int row_number = NewAdaThresImage.rows;
		int col_number = NewAdaThresImage.cols;
		
		for (int i = 0; i < row_number; i++)
		{
			for (int j = 0; j < col_number; j++)
			{
				if (ZerImageM00.at<double>(i, j) == 0)
					continue;

				//compute theta
				//vector<vector<double> > theta2(0);
				double theta_temporary = atan2(ZerImageM31I.at<double>(i, j), ZerImageM31R.at<double>(i, j));
				//theta2[i].push_back(thetaTem);

				//compute Z11'/Z31'
				double rotated_z11 = 0.0;
				rotated_z11 = sin(theta_temporary) * (ZerImageM11I.at<double>(i, j)) + cos(theta_temporary) * (ZerImageM11R.at<double>(i, j));
				double rotated_z31 = 0.0;
				rotated_z31 = sin(theta_temporary) * (ZerImageM31I.at<double>(i, j)) + cos(theta_temporary) * (ZerImageM31R.at<double>(i, j));

				//compute l
				double l_method1 = sqrt((5 * ZerImageM40.at<double>(i, j) + 3 * ZerImageM20.at<double>(i, j)) / (8 * ZerImageM20.at<double>(i, j)));
				double l_method2 = sqrt((5 * rotated_z31 + rotated_z11) / (6 * rotated_z11));
				double l = (l_method1 + l_method2) / 2;
				//compute k/h
				double k, h;

				k = 3 * rotated_z11 / 2 / pow((1 - l_method2 * l_method2), 1.5);
				h = (ZerImageM00.at<double>(i, j) - k * pi / 2 + k * asin(l_method2) + k * l_method2 * sqrt(1 - l_method2 * l_method2)) / pi;

				//judge the edge
				double k_value = 20.0;

				double l_value = sqrt(2) / GN;

				double absl = abs(l_method2 - l_method1);
				if (k >= k_value && absl <= l_value)
				{
					cv::Point2d point_temporary;
					point_temporary.x = j + GN * l * cos(theta_temporary) / 2;
					point_temporary.y = i + GN * l * sin(theta_temporary) / 2;
					SubEdgePoints.push_back(point_temporary);
				}
				else
				{
					continue;
				}
			}
		}
		
		// show subpixel
		for (size_t i = 0; i < SubEdgePoints.size(); i++)
		{
			x.push_back(SubEdgePoints[i].x); 
			y.push_back(SubEdgePoints[i].y);

			cv::Point center_forshow(cvRound(SubEdgePoints[i].x), cvRound(SubEdgePoints[i].y));
			cv::circle(mImage, center_forshow, 1, cv::Scalar(0, 97, 255), 1, 8, 0);
		}
		cv::namedWindow("subPixel", cv::WINDOW_FREERATIO);
		cv::imshow("subPixel", mImage);
		cv::imwrite("E:\\Code\\EllipseFitSource\\result\\subpixel.png", mImage);
		cv::waitKey(0);
	}

	void ZernikeMoment::MedianBlur(const cv::Mat& source, cv::Mat& target, const int& windows)
	{
		target = source.clone();
		const int Row = target.rows;
		const int Col = target.cols;
		
		if (Row < windows || Col < windows) return;
		const int halfWindow = windows / 2;

		for (int row = 0; row < Row-windows; row++)
		{
			for (int col = 0; col < Col-windows; col++)
			{
				double median = 0.0;
				MinHeap right; MaxHeap left;

				for (int i = row; i < row + windows; i++)
				{
					for (int j = col; j < col + windows; j++)
					{
						left.push(source.at<uchar>(i, j)); 
						right.push(left.top());
						left.pop();
						if (right.size() - left.size() > 1)
						{
							left.push(right.top());
							right.pop();
						}
					}
				}

				int size = left.size() + right.size();
				if (size % 2 == 0) median = (left.top() + right.top()) * 0.5;
				else median = right.top();

				target.at<uchar>(row + halfWindow, col + halfWindow) = median;
			}
		}
	}
}