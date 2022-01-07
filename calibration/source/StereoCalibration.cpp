#pragma once
#include "StereoCalibration.h"

using namespace std;
using namespace ceres;

//#define DEBUG

namespace MtStereoCalibration
{
	void StereoCalibration::compute()
	{
		// generate feature points
		const std::string id_left = "left";
		const std::string id_right = "right";
		
		GenerateImagePointsCV(mLeftImage, id_left, mLeftImagePoints);
		GenerateImagePointsCV(mRightImage, id_right, mRightImagePoints);
		/*GenerateImagePointsOwner(mLeftImage, id_left, mLeftImagePoints);
		GenerateImagePointsOwner(mRightImage, id_right, mRightImagePoints);*/

		MonocularCalibration(id_left);
		MonocularCalibration(id_right);

		assert(mLeftRListMat.size() == mImageNum);
		assert(mLefttListMat.size() == mImageNum);
		assert(mLeftRListMat.size() == mRightRListMat.size());
		assert(mLefttListMat.size() == mRighttListMat.size());

		mRListMat.resize(mImageNum);
		mtListMat.resize(mImageNum);

		for (int it = 0; it < mImageNum; it++)
		{
			mRListMat[it] = mRightRListMat[it] * mLeftRListMat[it].t();
			mtListMat[it] = mRighttListMat[it] - mRListMat[it] * mLefttListMat[it];
		}

		// rectification
		for (int it = 0; it < mImageNum; it++)
		{
			Eigen::Matrix3d RLeft, RRight;
			cv::Mat leftImageRectification, rightImageRectification;
			ComputeRectificationParameter(mRListMat[it],mtListMat[it],RLeft,RRight);
			Rectification(mLeftImage[it], mRightImage[it], RLeft, RRight, leftImageRectification, rightImageRectification);
		}

		ComputeReprojectionError(mRListMat, mtListMat);
	}

	void StereoCalibration::GenerateObjectPointsOwner(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<std::vector<Eigen::Vector3d>>& objectPoints)
	{
		std::vector<Eigen::Vector3d> objectPoint;
		for (int row = 0; row < borderSize.width; ++row)
		{
			for (int col = 0; col < borderSize.height; ++col) {

				objectPoint.push_back(Eigen::Vector3d(col * squareSize.height, row * squareSize.width, 0.0));
			}
		}

		for (int it = 0; it < mImageNum; it++)
		{
			objectPoints.push_back(objectPoint);
		}
	}
	void StereoCalibration::GenerateObjectPointsCV(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<std::vector<Eigen::Vector3d>>& objectPoints)
	{
		std::vector<Eigen::Vector3d> objectPoint;
		for (int row = 0; row < borderSize.height; ++row)
		{
			for (int col = 0; col < borderSize.width; ++col) {

				objectPoint.push_back(Eigen::Vector3d(col * squareSize.width, row * squareSize.height, 0.0));
			}
		}

		for (int it = 0; it < mImageNum; it++)
		{
			objectPoints.push_back(objectPoint);
		}
	}

	void StereoCalibration::GenerateImagePointsOwner(const std::vector<cv::Mat>& images, const std::string& id, std::vector<std::vector<Eigen::Vector2d>>& imagePoints)
	{
		int index = 1;
		for (auto img : images)
		{
			std::vector<cv::Point2d> corners;
			cv::Mat gray; cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
			FeatureDetector detect(img, 9, 11);
			detect.compute(corners);

			std::cout << "Extract " << id << " " << index << " Image's Corners!" << std::endl;

			std::vector<Eigen::Vector2d> _corners;
			for (auto corner : corners)
			{
				_corners.emplace_back(corner.x, corner.y);
			}
			imagePoints.push_back(_corners);
			index++;
		}
		
	}

	void StereoCalibration::GenerateImagePointsCV(const std::vector<cv::Mat>& images, const std::string& id, std::vector<std::vector<Eigen::Vector2d>>& imagePoints)
	{
		int index = 1;
		for (auto img : images)
		{
			std::vector<cv::Point2f> corners;
			cv::Mat gray; cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
			bool ok = cv::findChessboardCorners(gray, mBorderSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			if (ok)
			{
				std::cout << "Extract " << id << " " << index << " Image's Corners!" << std::endl;
				cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
			}

#ifdef DEBUG
			cv::drawChessboardCorners(img, mBorderSize, cv::Mat(corners), true);
			cv::namedWindow("corners" + to_string(index), cv::WINDOW_KEEPRATIO);
			cv::imshow("corners" + to_string(index), img);
			cv::waitKey(100);
#endif

			std::vector<Eigen::Vector2d> _corners;
			for (auto corner : corners)
			{
				_corners.emplace_back(corner.x, corner.y);
			}
			imagePoints.push_back(_corners);
			index++;
		}
	}

	void StereoCalibration::MonocularCalibration(const std::string& id)
	{
		std::string cameraParaPath = "E:\\Code\\EllipseFitSource\\config\\config.yaml";

		if (id == "left")
		{
			ZZYCalibration calibration(mDistortionParaNum, mLeftImage.size(), cameraParaPath);
			
			calibration.setObjectPoints(mObjectPoints);
			calibration.setImagePoints(mLeftImagePoints);

			calibration.compute(mLeftCameraMatrix, mLeftRadialDistortion, mLeftTangentialDistortion);
			calibration.getCameraPose(mLeftRListMat, mLefttListMat);
			calibration.get3DPoint(mLeftLandMark);
		}
		else if (id == "right")
		{
			ZZYCalibration calibration(mDistortionParaNum, mRightImage.size(), cameraParaPath);

			calibration.setObjectPoints(mObjectPoints);
			calibration.setImagePoints(mRightImagePoints);

			std::vector<std::vector<Eigen::Vector3d>> vX3D;
			calibration.compute(mRightCameraMatrix, mRightRadialDistortion, mRightTangentialDistortion);
			calibration.getCameraPose(mRightRListMat, mRighttListMat);
			calibration.get3DPoint(mRightLandMark);
		}
	}

	void StereoCalibration::ComputeReprojectionError(const std::vector<cv::Mat>& R, const std::vector<cv::Mat>& t)
	{
		const int leftFeaturePointNums = mImageNum * mLeftLandMark[0].size();
		const int rightFeaturePointNums = mImageNum * mRightLandMark[0].size();

		assert(leftFeaturePointNums == rightFeaturePointNums);

		for (int it = 0; it < mImageNum; it++)
		{
			double reproj_all = 0.;
			for (int i = 0; i < mImageNum; i++)
			{
				//std::vector<Eigen::Vector3d> LandMarkEveryImage;
				for (int j = 0; j < mLeftLandMark[i].size(); j++)
				{
					Eigen::Matrix3d _R; cv::cv2eigen(R[it], _R);
					Eigen::Vector3d _t; cv::cv2eigen(t[it], _t);

					auto Pr = mRightLandMark[i][j];
					auto Pl = mLeftLandMark[i][j];

					auto Pr_measure = _R * Pl + _t;
					//LandMarkEveryImage.emplace_back(Pr_measure(0), Pr_measure(1), Pr_measure(2));
					double err = (Pr_measure - Pr).norm();

					reproj_all += err;
				}
				//mRightLandMarkMeasure.push_back(LandMarkEveryImage);
			}
			std::cout << it << " reprojection error: " << reproj_all / leftFeaturePointNums << std::endl;
		}

		// get minimize reprojection error's camera pose
		for (int i = 0; i < mImageNum; i++)
		{
			std::vector<Eigen::Vector3d> LandMarkEveryImage;
			for (int j = 0; j < mLeftLandMark[i].size(); j++)
			{
				Eigen::Matrix3d _R; cv::cv2eigen(R[0], _R);
				Eigen::Vector3d _t; cv::cv2eigen(t[0], _t);

				auto Pr = mRightLandMark[i][j];
				auto Pl = mLeftLandMark[i][j];

				auto Pr_measure = _R * Pl + _t;
				LandMarkEveryImage.emplace_back(Pr_measure(0), Pr_measure(1), Pr_measure(2));
			}
			mRightLandMarkMeasure.push_back(LandMarkEveryImage);
		}
	}

	void StereoCalibration::ComputeRectificationParameter(const cv::Mat& R, const cv::Mat& t, Eigen::Matrix3d& RLeft, Eigen::Matrix3d& RRight)
	{
		const cv::Mat extrinsicR = R;
		const cv::Mat extrinsict = t;
		
		Eigen::Matrix3d R_; cv::cv2eigen(extrinsicR, R_);
		Eigen::Vector3d t_; cv::cv2eigen(extrinsict, t_);

		Eigen::AngleAxisd rotationVector(R_);
		Eigen::AngleAxisd rotationVectorLeft(rotationVector.angle() * 0.5, rotationVector.axis());
		Eigen::AngleAxisd rotationVectorRight(-rotationVector.angle() * 0.5, rotationVector.axis());

		RLeft = rotationVectorLeft.matrix();
		RRight = rotationVectorRight.matrix();

		Eigen::Vector3d e1(t_(0), t_(1), t_(2)); e1 = e1/e1.norm();
		Eigen::Vector3d e2(t_(1), -t_(0), 0); e2 = e2 / std::sqrt(t_(0) * t_(0) + t_(1) * t_(1));
		Eigen::Vector3d e3 = e1.cross(e2); e3 = e3/e3.norm();

		Eigen::Matrix3d R_rect;
		R_rect.block(0, 0, 1, 3) = e1.transpose();
		R_rect.block(1, 0, 1, 3) = e2.transpose();
		R_rect.block(2, 0, 1, 3) = e3.transpose();

		RLeft = R_rect * RLeft;
		RRight = R_rect * RRight;
	}

	void StereoCalibration::Rectification(const cv::Mat& leftImage, const cv::Mat& rightImage,
										  const Eigen::Matrix3d& RLeft, const Eigen::Matrix3d& RRight,
										  cv::Mat& leftImageRectification, cv::Mat& rightImageRectification)
	{
		const double fx_left = mLeftCameraMatrix(0, 0);
		const double fy_left = mLeftCameraMatrix(1, 1);
		const double cx_left = mLeftCameraMatrix(0, 2);
		const double cy_left = mLeftCameraMatrix(1, 2);
		
		const double fx_right = mRightCameraMatrix(0, 0);
		const double fy_right = mRightCameraMatrix(1, 1);
		const double cx_right = mRightCameraMatrix(0, 2);
		const double cy_right = mRightCameraMatrix(1, 2);

		const double fx_co = 0.5 * (fx_left + fx_right);
		const double fy_co = 0.5 * (fy_left + fy_right);
		const double cx_co = 0.5 * (cx_left + cx_right);
		const double cy_co = 0.5 * (cy_left + cy_right);

		const double k11 = mLeftRadialDistortion(0);
		const double k12 = mLeftRadialDistortion(1);
		const double k13 = mLeftRadialDistortion(2);
		const double p11 = mLeftTangentialDistortion(0);
		const double p12 = mLeftTangentialDistortion(1);

		const double k21 = mRightRadialDistortion(0);
		const double k22 = mRightRadialDistortion(1);
		const double k23 = mRightRadialDistortion(2);
		const double p21 = mRightTangentialDistortion(0);
		const double p22 = mRightTangentialDistortion(1);

		assert(leftImage.rows == rightImage.rows);
		assert(leftImage.cols == rightImage.cols);
		const int Row = leftImage.rows;
		const int Col = leftImage.cols;

		cv::Mat leftImageGray; cv::cvtColor(leftImage, leftImageGray, cv::COLOR_RGB2GRAY);
		cv::Mat rightImageGray; cv::cvtColor(rightImage, rightImageGray, cv::COLOR_RGB2GRAY);

		leftImageRectification = cv::Mat(Row,Col,CV_8UC1);
		rightImageRectification = cv::Mat(Row,Col,CV_8UC1);;

		// left image
		for (int x = 0; x < Col; x++)
		{
			for (int y = 0; y < Row; y++)
			{
				double xx = (x - cx_co) / fx_co;
				double yy = (y - cy_co) / fy_co;

				double r = xx * xx + yy * yy;
				double xxx = xx * (1. + k11 * r + k12 * r*r + k13 * r*r*r) + 2 * p11 * yy + p12 * (r + 2 * xx * xx);
				double yyy = yy * (1. + k11 * r + k12 * r*r + k13 * r*r*r) + 2 * p12 * xx + p11 * (r + 2 * yy * yy);

				Eigen::Vector3d normalizedCor (xxx,yyy,1.);
				normalizedCor =  RLeft*normalizedCor; normalizedCor /= normalizedCor(2);

				xxx = normalizedCor(0);
				yyy = normalizedCor(1);

				double xxxx = xxx * fx_left + cx_left;
				double yyyy = yyy * fy_left + cy_left;

				if (xxxx >= 0 && xxxx < Col && yyyy >= 0 && yyyy < Row)
				{
					double h = yyyy;
					double w = xxxx;
					leftImageRectification.at<uchar>(y, x) = (std::floor(w + 1) - w) * (std::floor(h + 1) - h) * leftImageGray.at<uchar>(std::floor(h), std::floor(w)) +
															(std::floor(w + 1) - w) * (h - std::floor(h)) * leftImageGray.at<uchar>(std::floor(h + 1), std::floor(w)) +
															(w - std::floor(w)) * (std::floor(h + 1) - h) * leftImageGray.at<uchar>(std::floor(h), std::floor(w + 1)) +
															(w - std::floor(w)) * (h - std::floor(h)) * leftImageGray.at<uchar>(std::floor(h + 1), std::floor(w + 1));
				}
			}
		}

		// right image
		for (int x = 0; x < Col; x++)
		{
			for (int y = 0; y < Row; y++)
			{
				double xx = (x - cx_co) / fx_co;
				double yy = (y - cy_co) / fy_co;

				double r = xx * xx + yy * yy;
				double xxx = xx * (1. + k21 * r + k22 * r * r + k23 * r * r * r) + 2 * p21 * yy + p22 * (r + 2 * xx * xx);
				double yyy = yy * (1. + k21 * r + k22 * r * r + k23 * r * r * r) + 2 * p22 * xx + p21 * (r + 2 * yy * yy);

				Eigen::Vector3d normalizedCor (xxx,yyy,1.);
				normalizedCor =  RLeft*normalizedCor; normalizedCor /= normalizedCor(2);

				xxx = normalizedCor(0);
				yyy = normalizedCor(1);

				double xxxx = xxx * fx_right + cx_right;
				double yyyy = yyy * fy_right + cy_right;

				if (xxxx >= 0 && xxxx < Col && yyyy >= 0 && yyyy < Row)
				{
					double h = yyyy;
					double w = xxxx;
					rightImageRectification.at<uchar>(y, x) = (std::floor(w + 1) - w) * (std::floor(h + 1) - h) * rightImageGray.at<uchar>(std::floor(h), std::floor(w)) +
															(std::floor(w + 1) - w) * (h - std::floor(h)) * rightImageGray.at<uchar>(std::floor(h + 1), std::floor(w)) +
															(w - std::floor(w)) * (std::floor(h + 1) - h) * rightImageGray.at<uchar>(std::floor(h), std::floor(w + 1)) +
															(w - std::floor(w)) * (h - std::floor(h)) * rightImageGray.at<uchar>(std::floor(h + 1), std::floor(w + 1));
				}
			}
		}

		cv::Size sizeLeft = leftImageRectification.size();
		cv::Size sizeRight = rightImageRectification.size();
		cv::Mat imgLeftRightRectification (sizeLeft.height, sizeLeft.width+sizeRight.width, CV_8UC1);
		cv::Mat left(imgLeftRightRectification, cv::Rect(0, 0, sizeLeft.width, sizeLeft.height));
		leftImageRectification.copyTo(left);
		cv::Mat right(imgLeftRightRectification, cv::Rect(sizeLeft.width, 0, sizeRight.width, sizeRight.height));
		rightImageRectification.copyTo(right);

		cv::Size sizeLeft_ = leftImageGray.size();
		cv::Size sizeRight_ = rightImageGray.size();
		cv::Mat imgLeftRightGray(sizeLeft_.height, sizeLeft_.width + sizeRight_.width, CV_8UC1);
		cv::Mat left_(imgLeftRightGray, cv::Rect(0, 0, sizeLeft_.width, sizeLeft_.height));
		leftImageGray.copyTo(left_);
		cv::Mat right_(imgLeftRightGray, cv::Rect(sizeLeft_.width, 0, sizeRight_.width, sizeRight_.height));
		rightImageGray.copyTo(right_);

		const int x_ = imgLeftRightGray.cols;
		for (int it = 0; it < imgLeftRightGray.rows; it += 50)
		{
			cv::line(imgLeftRightGray, cv::Point(0, it), cv::Point(x_, it), cv::Scalar(0, 0, 255), 1);
		}
		cv::imshow("original", imgLeftRightGray);

		// plot line
		const int x = imgLeftRightRectification.cols;
		for (int it = 0; it < imgLeftRightRectification.rows; it += 50)
		{
			cv::line(imgLeftRightRectification, cv::Point(0, it), cv::Point(x, it), cv::Scalar(0, 0, 255), 1);
		}
		cv::imshow("rectification", imgLeftRightRectification);
		cv::waitKey(0);
	}
}