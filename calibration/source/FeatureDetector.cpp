#include"FeatureDetector.h"

#define RELEASE

namespace MtFeatureDetector
{
	FeatureDetectorFailure::FeatureDetectorFailure(const char* msg) :Message(msg) {};
	const char* FeatureDetectorFailure::GetMessage() { return Message; }

	void InitialImage(cv::Mat& image, const int val)
	{
		const int Row = image.rows;
		const int Col = image.cols;
		
		for (int row = 0; row < Row; row++)
		{
			for (int col = 0; col < Col; col++)
			{
				image.at<uchar>(row, col) = val;
			}
		}
	}

	void FeatureDetector::compute(std::vector<cv::Point2d>& features)
	{
		int threshold = Otsu(mGrayImage);
		const int Row = mGrayImage.rows;
		const int Col = mGrayImage.cols;

		cv::Mat BinaryGraph(Row, Col, mGrayImage.type());
		for (int i = 0; i < Row; i++)
		{
			for (int j = 0; j < Col; j++)
			{
				if (mGrayImage.at<uchar>(i, j) < threshold)
					BinaryGraph.at<uchar>(i, j) = 0;
				else
					BinaryGraph.at<uchar>(i, j) = mGrayImage.at<uchar>(i, j);
			}
		}

		RegionDetect detector;
		MatrixP region;
		detector.setInputImage(BinaryGraph);
		detector.compute(region,threshold);

		MatrixP ellipseRegion;

		const int regionNum = region.size();
		
		std::vector<pair<int, double>> vCircleArea;
		vCircleArea.reserve(regionNum);

		for (int i = 0; i < regionNum; i++)
		{
			double area, ratioRec, ratioAll;
			minRectangleRatio(threshold, region[i],area, ratioRec,ratioAll);
			
			if (ratioRec >= 0.7 && ratioAll < 0.02)
			{
				if (region[i].first.rows > 20 && region[i].first.cols > 20)
				{
					double roundness = CheckRoundness(region[i].first);
					if (roundness > 0.85)
					{
						pair<int, double> tmp;
						tmp.second = area;
						tmp.first = ellipseRegion.size();
						vCircleArea.push_back(tmp);

						ellipseRegion.push_back(region[i]);
					}
				}
			}
 		}
		
		std::vector<cv::Point2d> vBenchMarkCircleFeatures;
		vBenchMarkCircleFeatures.reserve(5);
		{
			MaxHeap maxHeap;
			for (int i = 0; i < vCircleArea.size(); i++)
				maxHeap.push(vCircleArea[i]);

			int index1 = maxHeap.top().first; maxHeap.pop();
			int index2 = maxHeap.top().first; maxHeap.pop();
			int index3 = maxHeap.top().first; maxHeap.pop();
			int index4 = maxHeap.top().first; maxHeap.pop();
			int index5 = maxHeap.top().first; maxHeap.pop();

			for (int i = 0; i < ellipseRegion.size(); i++)
			{
				cv::Mat1d ellipseParameter;
				const int ellipseRoiStartX = ellipseRegion[i].second.x;
				const int ellipseRoiStartY = ellipseRegion[i].second.y;
				const int ellipseRoiRow = ellipseRegion[i].first.rows;
				const int ellipseRoiCol = ellipseRegion[i].first.cols;
				cv::Mat ellipseRoi = mGrayImage(cv::Rect(ellipseRoiStartX,
					ellipseRoiStartY, ellipseRoiCol, ellipseRoiRow)).clone();
				cv::GaussianBlur(ellipseRoi, ellipseRoi, cv::Size(3,3), 1.2, 1.2, 4);
				MtDualEllipseFitting::EllipseSolver ell(ellipseRoi);
				int state = ell.compute(true);
				if (state == 0)
				{
					ell.getConicPara(ellipseParameter);

					cv::Point2d feature;
					feature.x = ellipseParameter(0, 0) + ellipseRoiStartX;
					feature.y = ellipseParameter(1, 0) + ellipseRoiStartY;
					features.push_back(feature);

					if (i == index1 || i == index2 || i == index3 || i == index4 || i == index5)
						vBenchMarkCircleFeatures.push_back(feature);

					cv::ellipse(mImage, cv::Point(feature.x, feature.y),
						cv::Size(ellipseParameter(2, 0), ellipseParameter(3, 0)),
						ellipseParameter(4, 0), 0, 360, cv::Scalar(0, 0, 255), 1, cv::LINE_8);

					//cv::imshow("ellipsefit", mImage);
					
				}
				else
					return;
			}
			//cv::waitKey(0);
			cv::imwrite("E:\\Code\\EllipseFitSource\\result\\save.jpg", mImage);
		}
		
 		cv::Point2d pBenchMarkFeature1, pBenchMarkFeature2;
		GetClosetTwoPoint(vBenchMarkCircleFeatures, pBenchMarkFeature1, pBenchMarkFeature2);
		SortFeatures(features,pBenchMarkFeature1,pBenchMarkFeature2, mRow, mCol);
	}

	void FeatureDetector::GetClosetTwoPoint(const std::vector<cv::Point2d>& features, cv::Point2d& p1, cv::Point2d& p2)
	{
		const int number = features.size();
		
		pair<int, int> minIndex(-1,-1);
		double minDis = FLT_MAX;

		for (int i = 0; i < number-1; i++)
		{
			for (int j = i+1; j < number; j++)
			{
				double dis = computeDistance2D(features[i],features[j]);
				if (dis < minDis)
				{
					minIndex.first = i;
					minIndex.second = j;
					minDis = dis;
				}
			}
		}

		p1 = features[minIndex.first];
		p2 = features[minIndex.second];

	}
		

	void FeatureDetector::GetCentroid(const std::vector<cv::Point2d>& features, cv::Point2d& gravity)
	{
		const int number = features.size();
		double nInv = 1.0 / number;

		double x = 0.0;
		double y = 0.0;
		
		for (int i = 0; i < number; i++)
		{
			x += features[i].x; 
			y += features[i].y;
		}

		gravity.x = x * nInv;
		gravity.y = y * nInv;
	}

	void AddDrift(std::vector<cv::Point2d>& features)
	{
		float x_min = FLT_MAX;
		float y_min = FLT_MAX;
		
		const int number = features.size();
		
		for (int i = 0; i < number; i++)
		{
			if (features[i].x < x_min)
				x_min = features[i].x;
			if (features[i].y < y_min)
				y_min = features[i].y;
		}

		for (int i = 0; i < number; i++)
		{
			features[i].x -= x_min;
			features[i].y -= y_min;
		}
	}

	void FeatureDetector::GetCornerByPoint(const std::vector<cv::Point2d>& features, const cv::Point2d& mark, cv::Point2d& feature)
	{
		const int number = features.size();
		std::vector<pair<int, double>> vDistance;
		vDistance.reserve(number);

		for (int i = 0; i < number; i++)
		{
			pair<int, double> distance;
			double dis = computeDistance2D(mark, features[i]);

			distance.first = i;
			distance.second = dis;
			vDistance.push_back(distance);
		}

		MaxHeap heap;
		for (int i = 0; i < number; i++)
			heap.push(vDistance[i]);

		feature = features[heap.top().first];
	}

	void FeatureDetector::RotateFeatures(const std::vector<cv::Point2d>& benchmark, 
		const std::vector<cv::Point2d>& features, 
		std::vector<cv::Point2d>& sortedFeatures)throw(FeatureDetectorFailure)
	{
		if (benchmark.size() != 4)
			throw FeatureDetectorFailure("BechMark Error!");
	
		const cv::Point2d p0 = benchmark[0];
		const cv::Point2d p1 = benchmark[1];
		const cv::Point2d p2 = benchmark[2];
		const cv::Point2d p3 = benchmark[3];

		double k1 = std::atan((p3.y - p0.y) / (p3.x - p0.x));
		double k2 = std::atan((p2.y - p1.y) / (p2.x - p1.x));
		double kMeans = (k1 + k2) / 2;

		double r11 = ceres::cos(kMeans);
		double r12 = ceres::sin(kMeans);
		double r21 = -ceres::sin(kMeans);
		double r22 = ceres::cos(kMeans);
		
		cv::Point2d gravity;
		GetCentroid(features, gravity);

		const int number = features.size();
		for (int i = 0; i < number; i++)
		{
			cv::Point2d sortedFeature;
			const float x = features[i].x;
			const float y = features[i].y;
			sortedFeature.x = r11 * (x - gravity.x) + r12 * (y - gravity.y);
			sortedFeature.y = r21 * (x - gravity.x) + r22 * (y - gravity.y);

			sortedFeature.x += gravity.x;
			sortedFeature.y += gravity.y;

			sortedFeatures.push_back(sortedFeature);
		}
	}

	void FeatureDetector::SortFeatures(std::vector<cv::Point2d>& features, const cv::Point2d& p1, const cv::Point2d& p2, const int& row, const int& col)
	{
		cv::Point2d gravity;
		GetCentroid(features, gravity);
		const int number = features.size();

		cv::Point2d boundary1;
		cv::Point2d boundary2;
		GetCornerByPoint(features, gravity, boundary1);
		GetCornerByPoint(features, boundary1, boundary2);

		const double x1 = boundary1.x;
		const double y1 = boundary1.y;
		const double x2 = boundary2.x;
		const double y2 = boundary2.y;
		
		const double a = (y2-y1)/(x2-x1);
		const double b = y1-a*x1;

		std::vector<pair<int, double>> vDistance;
		vDistance.reserve(number);

		for (int i = 0; i < number; i++)
		{
			pair<int, double> distance;
			cv::Point2d point = features[i];
			double dis = computeDistance2D(point, a, b);

			distance.first = i;
			distance.second = dis;
			vDistance.push_back(distance);
		}

		MaxHeap heap;
		for (int i = 0; i < number; i++)
			heap.push(vDistance[i]);

		cv::Point2d boundary3 = features[heap.top().first];
		heap.pop();
		cv::Point2d boundary4 = features[heap.top().first];
		heap.pop();

		std::vector<cv::Point2d> vCorners;
		std::vector<cv::Point2d> vSortedCorners;

		vCorners.push_back(boundary1);
		vCorners.push_back(boundary2);
		vCorners.push_back(boundary3);
		vCorners.push_back(boundary4);

		SortWithPolarCoordinate(vCorners, vSortedCorners);
		CheckOrder(features, vSortedCorners);

		/*for (int i = 0; i < vSortedCorners.size(); i++)
		{
			cv::Point2f feature = vSortedCorners[i];
			cv::line(mImage,feature , feature, cv::Scalar(0, 0, 255), 6, 8, 0);
			cv::namedWindow("plot", cv::WINDOW_FREERATIO);
			cv::imshow("plot", mImage);
			cv::waitKey(0);
		}*/

		// construct eight point

		std::vector<cv::Point2d> vSortedFeatures;
		vSortedFeatures.reserve(number);

		if (false)
		{
			Eigen::Matrix3d homography = Eigen::Matrix3d::Identity();
			FindHomography(vSortedCorners, homography);
			Eigen::Matrix3d HInv = homography.inverse();

			for (int i = 0; i < number; i++)
			{
				Eigen::Vector3d p1;
				p1 << features[i].x, features[i].y, 1.0;
				Eigen::Vector3d p2 = homography * p1;

				cv::Point2d point;
				point.x = p2(0) / p2(2);
				point.y = p2(1) / p2(2);
				vSortedFeatures.push_back(point);
			}
		}
		else
			RotateFeatures(vSortedCorners, features, vSortedFeatures);
		
		
		AddDrift(vSortedFeatures);

		std::vector<pair<cv::Point2d, cv::Point2d>> vpFeatures;
		for (int i = 0; i < number; i++)
		{
			pair<cv::Point2d, cv::Point2d> feature;
			feature.first = vSortedFeatures[i];
			feature.second = features[i];
			vpFeatures.push_back(feature);
		}

		/*for (int i = 0; i < number; i++)
		{
			cv::line(mImage, vpFeatures[i].first, vpFeatures[i].first, cv::Scalar(0, 255, 0), 5, 8, 0);
			cv::line(mImage, vpFeatures[i].second, vpFeatures[i].second, cv::Scalar(0, 0, 255), 5, 8, 0);
			cv::namedWindow("plot", cv::WINDOW_NORMAL);
			cv::imshow("plot", mImage);
			cv::waitKey(0);
		}
		cv::waitKey(10);*/
		
		/*cv::line(mImage, p1, p2, cv::Scalar(0, 0, 255), 5, 8, 0);
		cv::namedWindow("plot", cv::WINDOW_NORMAL);
		cv::imshow("plot", mImage);
		cv::waitKey(0);*/

		std::vector<cv::Point2d> SortedFeatures;
		SortWithXYAxis(vpFeatures,p1,p2, mRow, mCol, SortedFeatures);
		features = SortedFeatures;
	}

	void FeatureDetector::CheckDirection(std::vector<std::vector<pair<cv::Point2d,cv::Point2d>>>& features, const cv::Point2d& p1, const cv::Point2d& p2)
	{
		const int Row = features.size();
		const int Col = features[0].size();
		const cv::Point2d benchmark(0.5*(p1.x+p2.x),0.5*(p1.y+p2.y));

		const cv::Point2d p01 = features[0][0].second;
		const cv::Point2d p02 = features[0][Col - 1].second;
		const cv::Point2d p03 = features[Row - 1][0].second;
		const cv::Point2d p04 = features[Row - 1][Col - 1].second;

		double dMin = FLT_MAX;
		
		double d1 = computeDistance2D(benchmark, p01); if (d1 < dMin) dMin = d1;
		double d2 = computeDistance2D(benchmark, p02); if (d2 < dMin) dMin = d2;
		double d3 = computeDistance2D(benchmark, p03); if (d3 < dMin) dMin = d3;
		double d4 = computeDistance2D(benchmark, p04); if (d4 < dMin) dMin = d4;

		if (d1 == dMin)
		{
			return;
		}
		else if (d2 == dMin)
		{
			for (int i = 0; i < Row; i++)
			{
				const int mid = Col / 2;
				for (int j = 0; j < mid; j++)
				{
					pair<cv::Point2d,cv::Point2d> tmp;
					tmp = features[i][j];
					features[i][j] = features[i][Col - 1 - j];
					features[i][Col - 1 - j] = tmp;
				}
			}
		}
		else if (d3 == dMin)
		{
			const int mid = Row / 2;
			for (int i = 0; i < mid; i++)
			{
				std::vector<pair<cv::Point2d, cv::Point2d>> tmp;
				tmp = features[i];
				features[i] = features[Row - 1 - i];
				features[Row - 1 - i] = tmp;
			}
		}
		else if (d4 == dMin)
		{
			const int midr = Row / 2;
			for (int i = 0; i < midr; i++)
			{
				std::vector<pair<cv::Point2d, cv::Point2d>> tmp;
				tmp = features[i];
				features[i] = features[Row - 1 - i];
				features[Row - 1 - i] = tmp;
			}

			for (int i = 0; i < Row; i++)
			{
				const int midc = Col / 2;
				for (int j = 0; j < midc; j++)
				{
					pair<cv::Point2d, cv::Point2d> tmp;
					tmp = features[i][j];
					features[i][j] = features[i][Col - 1 - j];
					features[i][Col - 1 - j] = tmp;
				}
			}
		}
	}

	void BubbleSort(std::vector<pair<cv::Point2d,cv::Point2d>>& features)
	{
		const int number = features.size();
		
		for (int i = 0; i < number-1; i++)
		{
			for (int j = i + 1; j < number; j++)
			{
				if (features[i].first.x > features[j].first.x)
				{
					pair<cv::Point2d,cv::Point2d> tmp = features[i];
					features[i] = features[j];
					features[j] = tmp;
				}
			}
		}
	}

	void FeatureDetector::SortWithXYAxis(const std::vector<pair<cv::Point2d,cv::Point2d>>& features, const cv::Point2d& p1, const cv::Point2d& p2,
										const int& row, const int& col, std::vector<cv::Point2d>& sortedFeatures)throw(FeatureDetectorFailure)
	{
		std::vector<std::vector<pair<cv::Point2d,cv::Point2d>>> YAxis;
		const int number = features.size();
	
		float y_min = FLT_MAX;
		float y_max = -FLT_MIN;
		
		for (int i = 0; i < number; i++)
		{
			if (features[i].first.y > y_max)
				y_max = features[i].first.y;
			if (features[i].first.y < y_min)
				y_min = features[i].first.y;
		}

		float interval = (y_max-y_min)/(row - 1);

		YAxis.resize(row);
		for (int i = 0; i < number; i++)
		{
			int index = std::round(std::abs(features[i].first.y) / interval);
			YAxis[index].push_back(features[i]);
		}

		for (int i = 0; i < row; i++)
			BubbleSort(YAxis[i]);

		CheckDirection(YAxis, p1, p2);

		for (int i = 0; i < row; i++)
		{
			if (YAxis[i].size() != col)
				throw FeatureDetectorFailure("Row And Col Error!");

			for (int j = 0; j < col; j++)
				sortedFeatures.push_back(YAxis[i][j].second);
		}
		
		
	}

	Eigen::VectorXd FeatureDetector::solveHomographyDLT(const Eigen::MatrixXd& srcPoints, const Eigen::MatrixXd& dstPoints)throw(FeatureDetectorFailure)
	{
		if (srcPoints.rows() != dstPoints.rows())
		{
			throw FeatureDetectorFailure("Source feature points is not equal with Target feature points!");
		}
		// step 1
		const int number = srcPoints.rows();
		Eigen::MatrixXd coeffient(2 * number, 9);
		for (int i = 0; i < number; i++)
		{
			coeffient(2 * i, 0) = 0.0;
			coeffient(2 * i, 1) = 0.0;
			coeffient(2 * i, 2) = 0.0;
			coeffient(2 * i, 3) = dstPoints(i, 0);
			coeffient(2 * i, 4) = dstPoints(i, 1);
			coeffient(2 * i, 5) = 1.0;
			coeffient(2 * i, 6) = -dstPoints(i, 0) * srcPoints(i, 1);
			coeffient(2 * i, 7) = -srcPoints(i, 1) * dstPoints(i, 1);
			coeffient(2 * i, 8) = srcPoints(i, 1);

			coeffient(2 * i + 1, 0) = dstPoints(i, 0);
			coeffient(2 * i + 1, 1) = dstPoints(i, 1);
			coeffient(2 * i + 1, 2) = 1.0;
			coeffient(2 * i + 1, 3) = 0.0;
			coeffient(2 * i + 1, 4) = 0.0;
			coeffient(2 * i + 1, 5) = 0.0;
			coeffient(2 * i + 1, 6) = -srcPoints(i, 0) * dstPoints(i, 0);
			coeffient(2 * i + 1, 7) = -dstPoints(i, 1) * srcPoints(i, 0);
			coeffient(2 * i + 1, 8) = srcPoints(i, 0);
		}
		// step 2 
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(coeffient, ComputeThinU | ComputeThinV);
		Eigen::MatrixXd V = svd.matrixV();

		return V.rightCols(1);
	}

	bool FeatureDetector::Normalize(Eigen::MatrixXd& P, Eigen::Matrix3d& T)
	{
		double cx = P.col(0).mean();
		double cy = P.col(1).mean();

		P.array().col(0) -= cx;
		P.array().col(1) -= cy;

		double stdx = ceres::sqrt((P.col(0).transpose() * P.col(0)).mean());
		double stdy = ceres::sqrt((P.col(1).transpose() * P.col(1)).mean());

		double sqrt_2 = sqrt(2);
		double scalex = sqrt_2 / stdx;
		double scaley = sqrt_2 / stdy;

		P.array().col(0) *= scalex;
		P.array().col(1) *= scalex;

		T << scalex, 0, -scalex * cx,
			0, scaley, -scaley * cy,
			0, 0, 1;
		return true;
	}

	void FeatureDetector::FindHomography(const std::vector<cv::Point2d>& features, Eigen::Matrix3d& homography)
	{
		std::vector<cv::Point2d> vHomography;
		vHomography.resize(8);
		
		vHomography[0] = features[0];
		vHomography[2] = features[1];
		vHomography[4] = features[2];
		vHomography[6] = features[3];

		cv::Point2d feature1(0.5*(features[0].x+features[1].x),0.5*(features[0].y+features[1].y));
		cv::Point2d feature3(0.5*(features[1].x+features[2].x),0.5*(features[1].y+features[2].y));
		cv::Point2d feature5(0.5*(features[2].x+features[3].x),0.5*(features[2].y+features[3].y));
		cv::Point2d feature7(0.5*(features[3].x+features[0].x),0.5*(features[3].y+features[0].y));

		vHomography[1] = feature1;
		vHomography[3] = feature3;
		vHomography[5] = feature5;
		vHomography[7] = feature7;

		for (int i = 0; i < 8; i++)
		{
			cv::Point2d feature = vHomography[i];
			cv::line(mImage, feature, feature, cv::Scalar(0, 0, 255), 5, 8, 0);

			cv::namedWindow("plot", cv::WINDOW_FREERATIO);
			cv::imshow("plot", mImage);
			cv::waitKey(100);
		}

		// construct benchmark
		std::vector<cv::Point2d> vMapPoint;
		vMapPoint.resize(8);

		vMapPoint[0] = cv::Point2d(0, 0);
		vMapPoint[6] = cv::Point2d((mRow - 1) * 10, 0);
		vMapPoint[4] = cv::Point2d((mRow - 1) * 10, (mCol - 1) * 10);
		vMapPoint[2] = cv::Point2d(0, (mCol - 1) * 10);

		vMapPoint[1] = cv::Point2d((vMapPoint[0].x+vMapPoint[2].x)*0.5, (vMapPoint[0].y+vMapPoint[2].y)*0.5);
		vMapPoint[3] = cv::Point2d((vMapPoint[2].x+vMapPoint[4].x)*0.5, (vMapPoint[2].y+vMapPoint[4].y)*0.5);
		vMapPoint[5] = cv::Point2d((vMapPoint[4].x+vMapPoint[6].x)*0.5, (vMapPoint[4].y+vMapPoint[6].y)*0.5);
		vMapPoint[7] = cv::Point2d((vMapPoint[6].x+vMapPoint[0].x)*0.5, (vMapPoint[6].y+vMapPoint[0].y)*0.5);

		for (int i = 0; i < 8; i++)
		{
			cv::Point2d feature = vMapPoint[i];
			cv::line(mImage, feature, feature, cv::Scalar(0, 0, 255), 5, 8, 0);

			cv::namedWindow("plot", cv::WINDOW_FREERATIO);
			cv::imshow("plot", mImage);
			cv::waitKey(100);        
		}

		Eigen::MatrixXd srcMatrix(8, 3);
		Eigen::MatrixXd dstMatrix(8, 3);

		for (int i = 0; i < 8; i++)
		{
			srcMatrix(i, 0) = vHomography[i].x;
			srcMatrix(i, 1) = vHomography[i].y;
			srcMatrix(i, 2) = 1.0;

			dstMatrix(i, 0) = vMapPoint[i].x;
			dstMatrix(i, 1) = vMapPoint[i].y;
			dstMatrix(i, 2) = 1.0;
		}

		Eigen::Matrix3d srcT, dstT;
		Normalize(srcMatrix, srcT);
		Normalize(dstMatrix, dstT);
		
		Eigen::VectorXd v = solveHomographyDLT(srcMatrix, dstMatrix);

		{
			ceres::Problem optimizationProblem;
			for (int i = 0; i < 8; i++)
			{
				optimizationProblem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<HomographyCost, 1, 9>(new HomographyCost(srcMatrix(i, 0), srcMatrix(i, 1), dstMatrix(i, 0), dstMatrix(i, 1))),
					nullptr,
					v.data()
				);
			}

			ceres::Solver::Options options;
			options.minimizer_progress_to_stdout = false;
			options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
			ceres::Solver::Summary summary;
			ceres::Solve(options, &optimizationProblem, &summary);

			//std::cout << summary.BriefReport() << std::endl;
		}

		homography << v(0), v(3), v(6),
				v(1), v(4), v(7),
				v(2), v(5), v(8);
		homography = dstT.inverse() * homography * srcT;
		homography.array() /= homography(8);
	}

	void reverse(std::vector<cv::Point2d>& features)
	{
		const int number = features.size();
		const int mid = number / 2;
		
		for (int i = 0; i < mid; i++)
		{
			cv::Point2d tmp;
			tmp = features[i];
			features[i] = features[number-1-i];
			features[number-1-i] = tmp;
		}
		
	}

	void  FeatureDetector::CheckOrder(const std::vector<cv::Point2d>& features, std::vector<cv::Point2d>& sortedFeatures)
	{
		const int number = features.size();

		if (mRow != mCol)
		{
			const cv::Point2d p0 = sortedFeatures[0];
			const cv::Point2d p1 = sortedFeatures[1];
			const cv::Point2d p3 = sortedFeatures[3];

			std::vector<pair<int, double>> vDis;
			for (int i = 0; i < number; i++)
			{
				pair<int, double> tmp;
				double dis = computeDistance2D(p0, features[i]);

				tmp.first = i;
				tmp.second = dis;
				vDis.push_back(tmp);
			}

			MinHeap minHeapDis;
			for (int i = 0; i < number; i++)
			{
				minHeapDis.push(vDis[i]);
			}

			minHeapDis.pop();
			const cv::Point2d pmin = features[minHeapDis.top().first];
			minHeapDis.pop();
			const cv::Point2d pminmin = features[minHeapDis.top().first];
			minHeapDis.pop();

			const double dmin = computeDistance2D(p0, pmin);
			const double dminmin = computeDistance2D(p0, pminmin);
			const double d01 = computeDistance2D(p0, p1);
			const double d03 = computeDistance2D(p0, p3);

			const Eigen::Vector2f v(pmin.x - p0.x, pmin.y - p0.y);
			const Eigen::Vector2f v01(p1.x - p0.x, p1.y - p0.y);
			const Eigen::Vector2f v03(p3.x - p0.x, p3.y - p0.y);

			double cos_theta01 = v.dot(v01) / (v.norm() * v01.norm());
			double cos_theta03 = v.dot(v03) / (v.norm() * v03.norm());

			int num01, num03;

			if (cos_theta01 > 0.5 && cos_theta03 < 0.5)
			{
				num01 = round(d01 / dmin);
				num03 = round(d03 / dminmin);
			}
			else if (cos_theta01 < 0.5 && cos_theta03 > 0.5)
			{
				num01 = round(d01 / dminmin);
				num03 = round(d03 / dmin);
			}
			else
				throw FeatureDetectorFailure("Error!");

			if (num01 > num03)
			{
				reverse(sortedFeatures);
				cv::Point2d p_last = sortedFeatures[sortedFeatures.size()-1];
				sortedFeatures.erase(sortedFeatures.end()-1);
				sortedFeatures.insert(sortedFeatures.begin(), p_last);
			}
		}
	}

	void FeatureDetector::SortWithPolarCoordinate(const std::vector<cv::Point2d>& features, std::vector<cv::Point2d>& sortedFeatures)throw(FeatureDetectorFailure)
	{
		typedef pair<int, double> PointAlpha;
		
		cv::Point2d gravity;
		GetCentroid(features, gravity);
		
		const int number = features.size();
		std::vector<PointAlpha> vPointAlpha;
		vPointAlpha.reserve(number);

		for (int i = 0; i < number; i++)
		{
			Eigen::RowVector2f x_vector(1, 0);
			PointAlpha pointAlpha{};

			pointAlpha.first = i;

			if (features[i].y > gravity.y)
			{
				// Ahpha = alpha
				float dx = features[i].x - gravity.x;
				float dy = features[i].y - gravity.y;

				Eigen::RowVector2f pointVector(dx, dy);
				float alpha = acosf(pointVector.dot(x_vector) / sqrt(powf(dx, 2) + powf(dy, 2)));

				if (features[i].x == gravity.x) pointAlpha.second = PI / 2;
				else pointAlpha.second = alpha;
			}
			else if (features[i].y < gravity.y)
			{
				// Ahpha = 2PI - alpha
				float dx = features[i].x - gravity.x;
				float dy = features[i].y - gravity.y;

				Eigen::RowVector2f pointVector(dx, dy);
				float alpha = 2 * PI - acosf(pointVector.dot(x_vector) / sqrt(powf(dx, 2) + powf(dy, 2)));

				if (features[i].x == gravity.x) pointAlpha.second = 3 / 2 * PI;
				else pointAlpha.second = alpha;
			}
			else
			{
				if (features[i].x > gravity.x) pointAlpha.second = 0;
				else pointAlpha.second = PI;
			}
			vPointAlpha.push_back(pointAlpha);
		}

		// sort with maxmum heap
		MaxHeap heap;
		for (int i = 0; i < number; i++)
			heap.push(vPointAlpha[i]);
		
		std::vector<cv::Point2d> sorted;
		for (int i = 0; i < number; i++)
		{
			sorted.push_back(features[heap.top().first]);
			heap.pop();
		}
			

		cv::Point2d p0(0., 0.);
		double disMin = FLT_MAX;
		int indexMin = -1;
		
		for (int i = 0; i < number; i++)
		{
			double dis = computeDistance2D(p0,sorted[i]);
			if (dis < disMin)
			{
				disMin = dis;
				indexMin = i;
			}
		}

		sortedFeatures.clear();
		sortedFeatures.resize(number);
		for (int i = 0; i < number; i++)
			sortedFeatures[i] = sorted[(indexMin + i) % number];
		
	}

	void FeatureDetector::GrayMoment(const pair<cv::Mat,cv::Point2d>& image, float& u, float& v)throw(FeatureDetectorFailure)
	{
		const int Row = image.first.rows;
		const int Col = image.first.cols;

		int count = 0;
		double m10 = 0.0, m01 = 0.0, m00 = 0.0;
		
		for (int i = 0; i < Row; i++)
		{
			for (int j = 0; j < Col; j++)
			{
				if (image.first.at<uchar>(i, j) == 0)
					continue;
				m00 += image.first.at<uchar>(i, j);
				m10 += j * image.first.at<uchar>(i, j);
				m01 += i * image.first.at<uchar>(i, j);

				count++;
			}
		}

		if (count == 0)
			throw FeatureDetectorFailure("Input Image is a white background image!");

		u = m10 / m00;
		v = m01 / m00;

		u += image.second.x;
		v += image.second.y;
	}

	double FeatureDetector::computeDistance2D(const cv::Point2d& p1, const cv::Point2d& p2)
	{
		double dx = p2.x - p1.x;
		double dy = p2.y - p1.y;
		return ceres::sqrt(dx*dx+dy*dy);
	}

	double FeatureDetector::computeDistance2D(const cv::Point2d& p, const double& a, const double& b)
	{
		const double x = p.x;
		const double y = p.y;

		double dis = std::abs(a*x-y+b)/std::sqrt(a*a+1);
		return dis;
	}

	void FeatureDetector::calculateRectangleArea(const std::vector<cv::Point2d>& minRectangle, double& area)
	{
		const cv::Point2d v1 = minRectangle[0];
		const cv::Point2d v2 = minRectangle[1];
		const cv::Point2d v3 = minRectangle[2];
		const cv::Point2d v4 = minRectangle[3];
		
		double a1 = computeDistance2D(v1, v2)+1.0;
		double a2 = computeDistance2D(v3, v4)+1.0;
		
		double b1 = computeDistance2D(v1, v4)+1.0;
		double b2 = computeDistance2D(v2, v3)+1.0;

		area = 0.25 * (a1 + a2) * (b1 + b2);
	}

	double FeatureDetector::CheckRoundness(const cv::Mat& image)
	{
		cv::Mat grayImage;
		if (image.channels() == 3)
		{
			cv::cvtColor(image, grayImage, cv::COLOR_RGB2GRAY);
		}
		else if (image.channels() == 1)
		{
			grayImage = image.clone();
		}

#ifdef DEBUG
		cv::imshow("input", grayImage);
#endif
		
		// Gauss Blur
		cv::Mat blurImage;
		cv::GaussianBlur(grayImage, blurImage, cv::Size(15, 15), 0, 0);
		
#ifdef DEBUG
		cv::imshow("Gauss Blur", blurImage);
#endif

		// binary
		cv::Mat binaryImage;
		cv::threshold(blurImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);

#ifdef DEBUG
		cv::imshow("Binary Image", binaryImage);
#endif

		cv::Mat morphlogyImage;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
		cv::morphologyEx(binaryImage, morphlogyImage, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

#ifdef DEBUG
		imshow("close Image", morphlogyImage);
#endif

		cv::Mat resultImage = cv::Mat::zeros(image.size(), CV_8UC3);
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i>hierchy;
		cv::findContours(morphlogyImage, contours, hierchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());

		double area = 0.0;
		double lenth = 0.0;
		
		for (size_t t = 0; t < contours.size(); t++) {
			cv::Rect rect = cv::boundingRect(contours[t]);
			if (rect.width < image.cols / 2) continue;
			//if (rect.width > (image.cols - 20)) continue;
			area += cv::contourArea(contours[t]);
			lenth += cv::arcLength(contours[t], true);
			cv::drawContours(resultImage, contours, static_cast<int>(t), cv::Scalar(0, 0, 255), 2, 8, cv::Mat());
		}

#ifdef DEBUG
		cv::imshow("ResultImage", resultImage);
		cv::waitKey(0);
#endif
		double roundness = 4 * PI * area / (lenth * lenth);
		return roundness;
	}

	// 计算圆形椭圆占ROI区域的面积比率
	void FeatureDetector::minRectangleRatio(const int& threshold, const pair<cv::Mat,cv::Point2d>& image,double& area, double& ratioRec, double& ratioAll)
	{
		std::vector<cv::Point2f> vPoints;
		
		const int Row = image.first.rows;
		const int Col = image.first.cols;

		const int RowAll = mImage.rows;
		const int ColAll = mImage.cols;
		
		vPoints.reserve(Row * Col);
		double tarArea = 0.0;
		double allArea = RowAll*ColAll;
		
		for (int i = 0; i < Row; i++)
		{
			for (int j = 0; j < Col; j++)
			{
				if (image.first.at<uchar>(i, j) >= threshold)
				{
					cv::Point2f point;
					point.x = j;
					point.y = i;
					vPoints.push_back(point);

					tarArea += 1.0;
				}
			}
		}

		area = tarArea;

		cv::Point2f vertex[4];
		cv::RotatedRect minRect = minAreaRect(cv::Mat(vPoints));
		minRect.points(vertex);

		// draw the minimum area enclosing rectangle
		std::vector<cv::Point2d> minRectangle;
		cv::Mat plot = image.first.clone();
		for (int i = 0; i < 4; i++)
		{
			cv::line(plot, vertex[i], vertex[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, 8);
			minRectangle.push_back(vertex[i]);
		}

		/*cv::imshow("min rec", plot);
		cv::waitKey(0);*/

		// calculate area of minimum rectangle
		//double recArea = cv::contourArea(minRectangle);
		double recArea = 0.0;
		calculateRectangleArea(minRectangle, recArea);
		
		ratioRec = tarArea/recArea;
		ratioAll = tarArea/allArea;
	}

	int FeatureDetector::Otsu(const cv::Mat& image)
	{
		int th = 0;
		const int GrayScale = 256;					// Total Gray level of gray image
		int pixCount[GrayScale] = { 0 };			// The number of pixels occupied by each gray value
		int pixSum = image.rows * image.cols;		// The total number of pixels of the image
		float pixProportion[GrayScale] = { 0.0 };	// The proportion of total pixels occupied by each gray value

		float w0, w1, u0tmp, u1tmp, u0, u1, deltaTmp, deltaMax = 0;

		//std::ofstream out("E:\\Code\\EllipseFitSource\\data\\pixel.txt");

		// Count the number of pixels in each gray level
		for (int i = 0; i < image.cols; i++)
		{
			for (int j = 0; j < image.rows; j++)
			{
				int size = image.at<uchar>(j, i);
				pixCount[image.at<uchar>(j, i)]++;
				//out << size << " ";
			}
		}
		//out.close();

		for (int i = 0; i < GrayScale; i++)
		{
			pixProportion[i] = pixCount[i] * 1.0 / pixSum;
		}

		// Traverse the threshold segmentation conditions of all gray levels and test which one has the largest inter-class variance
		for (int i = 0; i < GrayScale; i++)
		{
			w0 = w1 = u0tmp = u1tmp = u0 = u1 = deltaTmp = 0;

			for (int j = 0; j < GrayScale; j++)
			{
				if (j <= i)		// background
				{
					w0 += pixProportion[j];
					u0tmp += j * pixProportion[j];
				}
				else			// foreground
				{
					w1 += pixProportion[j];
					u1tmp += j * pixProportion[j];
				}
			}
			u0 = u0tmp / w0;
			u1 = u1tmp / w1;

			// Between-class variance formula: g = w1 * w2 * (u1 - u2) ^ 2
			deltaTmp = (float)(w0 * w1 * pow((u0 - u1), 2));

			if (deltaTmp > deltaMax)
			{
				deltaMax = deltaTmp;
				th = i;
			}
		}
		return th;
	}

	void FeatureDetector::CalculateGradient(const cv::Mat& image, const GradientMethod& method)throw(FeatureDetectorFailure)
	{
		if (method != 1 && method != 2 && method != 3)
		{
			throw FeatureDetectorFailure("These Methods Are Not Set!");
			exit(-1);
		}

		cv::Mat GradientX;
		cv::Mat GradientY;

		if (method == GradientMethod::Sobel)
		{
			GradientX = SobelOperator(image, "x");
			GradientY = SobelOperator(image, "y");
		}
		else if (method == GradientMethod::Roberts)
		{
			GradientX = RobertsOperator(image, "x");
			GradientY = RobertsOperator(image, "y");
		}
		else if (method == GradientMethod::Prewitt)
		{
			GradientX = PrewittOperator(image, "x");
			GradientY = PrewittOperator(image, "y");
		}

		mGradientX = GradientX;
		mGradientY = GradientY;
	}

	// calculate gradient with subpixel thechnology
	/*cv::Mat BoundarySolver::SubPixelOperator(const cv::Mat& image, const string& flag)
	{
	
	}*/

	// calculate gradient with sobel operator
	cv::Mat FeatureDetector::SobelOperator(const cv::Mat& image, const string& flag)
	{
		cv::Mat gradient(image.rows, image.cols, image.type());
		if (flag == "x")
		{
			const int Row = image.rows;
			const int Col = image.cols;
			for (int row = 1; row < Row - 1; row++)
			{
				for (int col = 1; col < Col - 1; col++)
				{
					gradient.at<uchar>(row, col) = image.at<uchar>(row - 1, col + 1) + 2 * image.at<uchar>(row, col + 1) + image.at<uchar>(row + 1, col + 1)
						- image.at<uchar>(row - 1, col - 1) - 2 * image.at<uchar>(row, col - 1) - image.at<uchar>(row + 1, col - 1);
				}
			}
		}
		else if (flag == "y")
		{
			const int Row = image.rows;
			const int Col = image.cols;
			for (int row = 1; row < Row - 1; row++)
			{
				for (int col = 1; col < Col - 1; col++)
				{
					gradient.at<uchar>(row, col) = image.at<uchar>(row - 1, col - 1) + 2 * image.at<uchar>(row - 1, col) + image.at<uchar>(row - 1, col + 1)
						- image.at<uchar>(row + 1, col - 1) - 2 * image.at<uchar>(row + 1, col) - image.at<uchar>(row + 1, col + 1);
				}
			}
		}
		return gradient;
	}

	// calculate gradient with roberts operator
	cv::Mat FeatureDetector::RobertsOperator(const cv::Mat& image, const string& flag)
	{
		cv::Mat gradient(image.rows, image.cols, image.type());
		if (flag == "x")
		{
			const int Row = image.rows;
			const int Col = image.cols;
			for (int row = 0; row < Row - 1; row++)
			{
				for (int col = 0; col < Col - 1; col++)
				{
					gradient.at<uchar>(row, col) = image.at<uchar>(row, col) - image.at<uchar>(row + 1, col + 1);
				}
			}
		}
		else if (flag == "y")
		{
			const int Row = image.rows;
			const int Col = image.cols;
			for (int row = 0; row < Row - 1; row++)
			{
				for (int col = 0; col < Col - 1; col++)
				{
					gradient.at<uchar>(row, col) = image.at<uchar>(row + 1, col) - image.at<uchar>(row, col + 1);
				}
			}
		}
		return gradient;
	}

	// calculate gradient with prewitt operator
	cv::Mat FeatureDetector::PrewittOperator(const cv::Mat& image, const string& flag)
	{
		cv::Mat gradient(image.rows, image.cols, image.type());
		if (flag == "x")
		{
			const int Row = image.rows;
			const int Col = image.cols;
			for (int row = 1; row < Row - 1; row++)
			{
				for (int col = 1; col < Col - 1; col++)
				{
					gradient.at<uchar>(row, col) = image.at<uchar>(row - 1, col + 1) + image.at<uchar>(row, col + 1) + image.at<uchar>(row + 1, col + 1)
						- image.at<uchar>(row - 1, col - 1) - image.at<uchar>(row, col - 1) - image.at<uchar>(row + 1, col - 1);
				}
			}
		}
		else if (flag == "y")
		{
			const int Row = image.rows;
			const int Col = image.cols;
			for (int row = 1; row < Row - 1; row++)
			{
				for (int col = 1; col < Col - 1; col++)
				{
					gradient.at<uchar>(row, col) = image.at<uchar>(row - 1, col - 1) + image.at<uchar>(row - 1, col) + image.at<uchar>(row - 1, col + 1)
						- image.at<uchar>(row + 1, col - 1) - image.at<uchar>(row + 1, col) - image.at<uchar>(row + 1, col + 1);
				}
			}
		}
		return gradient;
	}
	
}

