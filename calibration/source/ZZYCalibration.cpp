#pragma once
#include "ZZYCalibration.h"
#include "CommonFunctions.h"

using namespace std;
using namespace ceres;

namespace MtZZYCalibration
{
	ZZYCalibrationFailure::ZZYCalibrationFailure(const char* msg) :Message(msg) {};
	const char* ZZYCalibrationFailure::GetMessage() { return Message; }

	void ZZYCalibration::compute(Eigen::Matrix3d& CameraMatrix, Eigen::Vector3d& RadialDistortion, Eigen::Vector2d& TangentialDistortion)
	{
		/*标定相机*/
		cv::Mat cameraMatrix;
		cv::Mat distCoeffs;
		Eigen::VectorXd distortionCoeffs;
		computeCameraCalibration(mImagePoints, mObjectPoints, cameraMatrix, distCoeffs);
		/*标定参数转换为Eigen格式*/
		cv::cv2eigen(cameraMatrix, CameraMatrix);
		cv::cv2eigen(distCoeffs, distortionCoeffs);
		/*解析畸变参数*/
		RadialDistortion(0) = distortionCoeffs(0);
		RadialDistortion(1) = distortionCoeffs(1);
		RadialDistortion(2) = distortionCoeffs(4);
		TangentialDistortion(0) = distortionCoeffs(2);
		TangentialDistortion(1) = distortionCoeffs(3);
		/*保存标定参数并打印*/
		mCameraMatrix = CameraMatrix;
		mRadialDistortion = RadialDistortion;
		mTangentialDistortion = TangentialDistortion;
		cv::FileStorage fs(mStrCameraParaPath.c_str(), cv::FileStorage::WRITE);
		fs << "Camera_fx" << CameraMatrix(0, 0);
		fs << "Camera_fy" << CameraMatrix(1, 1);
		fs << "Camera_cx" << CameraMatrix(0, 2);
		fs << "Camera_cy" << CameraMatrix(1, 2);
		fs << "Camera_k1" << RadialDistortion(0);
		fs << "Camera_k2" << RadialDistortion(1);
		fs << "Camera_k3" << RadialDistortion(2);
		fs << "Camera_p1" << TangentialDistortion(0);
		fs << "Camera_p2" << TangentialDistortion(1);
		fs << "Image_Rotation" << "[" << mRListMat << "]";
		fs << "Image_Translation" << "[" <<mtListMat << "]";
		this->PrintCameraIntrinsics();
	}

	void ZZYCalibration::PrintCameraIntrinsics()
	{
		std::cout << std::endl << "Camera Intrinsics" << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;
		std::cout << setiosflags(ios::left) << setw(14) << "Parameters" << resetiosflags(ios::left)
			<< setiosflags(ios::right) << setw(22) << "Value" << setw(10)
			<< resetiosflags(ios::right) << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;
		string parameters[] = { "FocalLength", "PrinciplePoint", "RadialDistortion", "TangentialDistortion" };
		string value[] = { "","","","" };
		value[0] = "[" + to_string((float)mCameraMatrix(0, 0)) + "," + to_string((float)mCameraMatrix(1, 1)) + "]";
		value[1] = "[" + to_string((float)mCameraMatrix(0, 2)) + "," + to_string((float)mCameraMatrix(1, 2)) + "]";
		value[2] = "[" + to_string((float)mRadialDistortion(0)) + "," + to_string((float)mRadialDistortion(1)) + "," + to_string((float)mRadialDistortion(2)) + "]";
		value[3] = "[" + to_string((float)mTangentialDistortion(0)) + "," + to_string((float)mTangentialDistortion(1)) + "]";
		for (int i = 0; i < 4; i++)
		{
			std::cout << setiosflags(ios::left) << setw(20) << parameters[i] << resetiosflags(ios::left)
				<< setiosflags(ios::right) << setw(25) << value[i] << setw(10) << resetiosflags(ios::right) << std::endl;
		}
		std::cout << "-------------------------------------------------" << std::endl;
	}

	void ZZYCalibration::Points2d2Vectors2d(const doublePoint2D& vIn, doubleVector2D& vOut)
	{
		for (int iti = 0; iti < vIn.size(); ++iti) {
			std::vector<Eigen::Vector2d> vEigenPts;
			for (int itj = 0; itj < vIn[iti].size();++itj) {
				Eigen::Vector2d eigenPt;
				CommonFunctions::Point2d2Vector2d(vIn[iti][itj], eigenPt);
				vEigenPts.push_back(eigenPt);
			}
			vOut.push_back(vEigenPts);
		}
		return;
	}

	void ZZYCalibration::Points3d2Vectors3d(const doublePoint3D& vIn, doubleVector3D& vOut)
	{
		for (int iti = 0; iti < vIn.size(); ++iti) {
			std::vector<Eigen::Vector3d> vEigenPts;
			for (int itj = 0; itj < vIn[iti].size();++itj) {
				Eigen::Vector3d eigenPt;
				CommonFunctions::Point3d2Vector3d(vIn[iti][itj], eigenPt);
				vEigenPts.push_back(eigenPt);
			}
			vOut.push_back(vEigenPts);
		}
		return;
	}

	bool ZZYCalibration::Normalize(const std::vector<Eigen::Vector2d>& vKeys, std::vector<Eigen::Vector2d>& vNormalizedPoints, Eigen::Matrix3d& T)
	{
		// step 1
		double meanX = 0.0;
		double meanY = 0.0;
		const int number = vKeys.size();
		for (int i = 0; i < number; i++)
		{
			meanX += vKeys[i](0, 0);
			meanY += vKeys[i](1, 0);
		}
		meanX /= (double)number;
		meanY /= (double)number;
		// step 2
		double meanDevX = 0.0;
		double meanDevY = 0.0;
		for (int i = 0; i < number; i++)
		{
			vNormalizedPoints[i](0, 0) = vKeys[i](0, 0) - meanX;
			vNormalizedPoints[i](1, 0) = vKeys[i](1, 0) - meanY;
			meanDevX += fabs(vNormalizedPoints[i](0, 0));
			meanDevY += fabs(vNormalizedPoints[i](1, 0));
		}
		meanDevX /= (double)number;
		meanDevY /= (double)number;
		for (int i = 0; i < number; i++)
		{
			vNormalizedPoints[i](0, 0) /= meanDevX;
			vNormalizedPoints[i](1, 0) /= meanDevY;
		}

		double sX = 1.0 / meanDevX;
		double sY = 1.0 / meanDevY;

		T = Eigen::Matrix3d::Identity(3, 3);
		T(0, 0) = sX;
		T(0, 2) = -meanX * sX;
		T(1, 1) = sY;
		T(1, 2) = -meanY * sY;
		T(2, 2) = 1;

		return true;
	}

	bool ZZYCalibration::Normalize(Eigen::MatrixXd& P, Eigen::Matrix3d& T)
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

	Eigen::VectorXd ZZYCalibration::solveHomographyDLT(const Eigen::MatrixXd& srcPoints, const Eigen::MatrixXd& dstPoints)throw(ZZYCalibrationFailure)
	{
		if (srcPoints.rows() != dstPoints.rows())
		{
			throw ZZYCalibrationFailure("Source feature points is not equal with Target feature points!");
		}
		// step 1
		const int number = srcPoints.rows();
		Eigen::MatrixXd coeffient(2 * number, 9);
		for (int i = 0; i < number; i++)
		{
			coeffient(2 * i, 0) = 0.0;
			coeffient(2 * i, 1) = 0.0;
			coeffient(2 * i, 2) = 0.0;
			coeffient(2 * i, 3) = srcPoints(i, 0);
			coeffient(2 * i, 4) = srcPoints(i, 1);
			coeffient(2 * i, 5) = 1.0;
			coeffient(2 * i, 6) = -srcPoints(i, 0) * dstPoints(i, 1);
			coeffient(2 * i, 7) = -srcPoints(i, 1) * dstPoints(i, 1);
			coeffient(2 * i, 8) = dstPoints(i, 1);

			coeffient(2 * i + 1, 0) = srcPoints(i, 0);
			coeffient(2 * i + 1, 1) = srcPoints(i, 1);
			coeffient(2 * i + 1, 2) = 1.0;
			coeffient(2 * i + 1, 3) = 0.0;
			coeffient(2 * i + 1, 4) = 0.0;
			coeffient(2 * i + 1, 5) = 0.0;
			coeffient(2 * i + 1, 6) = -srcPoints(i, 0) * dstPoints(i, 0);
			coeffient(2 * i + 1, 7) = -srcPoints(i, 1) * dstPoints(i, 0);
			coeffient(2 * i + 1, 8) = dstPoints(i, 0);
		}
		// step 2 
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(coeffient, ComputeThinU | ComputeThinV);
		Eigen::MatrixXd V = svd.matrixV();

		/*double s = V.rightCols(1)(8);
		MatrixXd M = V.rightCols(1) / s;*/

		return V.rightCols(1);
	}

	bool ZZYCalibration::findHomography(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H, bool isNormal)throw(ZZYCalibrationFailure)
	{
		if (srcPoints.size() != dstPoints.size())
		{
			throw ZZYCalibrationFailure("Source feature points is not equal with Target feature points!");
		}
		const int number = srcPoints.size();

		Eigen::Matrix3d srcT;
		Eigen::Matrix3d dstT;

		// step 1: normalize
		Eigen::MatrixXd srcMatrix(number, 3);
		Eigen::MatrixXd dstMatrix(number, 3);

		for (int i = 0; i < number; i++)
		{
			srcMatrix(i, 0) = srcPoints[i](0, 0);
			srcMatrix(i, 1) = srcPoints[i](1, 0);
			srcMatrix(i, 2) = 1.0;

			dstMatrix(i, 0) = dstPoints[i](0, 0);
			dstMatrix(i, 1) = dstPoints[i](1, 0);
			dstMatrix(i, 2) = 1.0;
		}

		if (isNormal)
		{
			Normalize(srcMatrix, srcT);
			Normalize(dstMatrix, dstT);
		}

		// step 2: DLT
		Eigen::VectorXd v = solveHomographyDLT(srcMatrix, dstMatrix);

		// step 3: optimization
		{
			ceres::Problem optimizationProblem;
			for (int i = 0; i < number; i++)
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

		// step 4: calculate H
		Eigen::Matrix3d M;
		M << v(0), v(3), v(6),
			v(1), v(4), v(7),
			v(2), v(5), v(8);

		if (isNormal)
		{
			H = dstT.inverse() * M * srcT;
			H.array() /= H(8);
		}
		else
		{
			H = M;
			H.array() /= H(8);
		}
		return true;
	}

	bool ZZYCalibration::findHomographyByOpenCV(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H)throw(ZZYCalibrationFailure)
	{
		if (srcPoints.size() != dstPoints.size())
		{
			throw ZZYCalibrationFailure("Source feature points is not equal with Target feature points!");
		}
		const int number = srcPoints.size();
		std::vector<cv::Point2f> objectPoints, imagePoints;

		for (int i = 0; i < number; i++)
		{
			objectPoints.push_back(cv::Point2f(srcPoints[i](0), srcPoints[i](1)));
			imagePoints.push_back(cv::Point2f(dstPoints[i](0), dstPoints[i](1)));
		}

		cv::Mat HMat = cv::findHomography(objectPoints, imagePoints, cv::RANSAC);
		cv::cv2eigen(HMat, H);

		return true;
	}

	Matrix3d ZZYCalibration::RotationVector2Matrix(const Eigen::Vector3d& v)
	{
		double s = std::sqrt(v.dot(v));
		Eigen::Vector3d axis = v / s;
		Eigen::AngleAxisd r(s, axis);
		return r.toRotationMatrix();
	}

	Eigen::Vector3d ZZYCalibration::RotationMatrix2Vector(const Eigen::Matrix3d& R)
	{
		Eigen::AngleAxisd r;
		r.fromRotationMatrix(R);
		return r.angle() * r.axis();
	}

	bool ZZYCalibration::findHomographyByRansac(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H, bool isNormal)throw(ZZYCalibrationFailure)
	{
		if (srcPoints.size() != dstPoints.size()) {
			throw ZZYCalibrationFailure("Source feature points is not equal with Target feature points!");
		}

		const int number = srcPoints.size();

		Eigen::Matrix3d srcT;
		Eigen::Matrix3d dstT;

		// step 1: normalize
		Eigen::MatrixXd srcMatrix(number, 3);
		Eigen::MatrixXd dstMatrix(number, 3);
		for (int i = 0; i < number; i++)
		{
			srcMatrix(i, 0) = srcPoints[i](0, 0);
			srcMatrix(i, 1) = srcPoints[i](1, 0);
			srcMatrix(i, 2) = 1.0;

			dstMatrix(i, 0) = dstPoints[i](0, 0);
			dstMatrix(i, 1) = dstPoints[i](1, 0);
			dstMatrix(i, 2) = 1.0;
		}
		if (isNormal)
		{
			Normalize(srcMatrix, srcT);
			Normalize(dstMatrix, dstT);
		}

		// step 2: ransac
		double p = 0.99, w = 0.5;
		int s = 4;
		int maxN = std::log(1 - p) / std::log(1 - pow((1 - w), s)) + 1;
		double threshold = 0.2;

		int bestCount = 0;
		std::vector<int> inlinersMask(number);
		for (int i = 0; i < maxN; i++)
		{
			cv::RNG rng(cv::getTickCount());
			std::set<int> indexs;
			while (indexs.size() < s)
			{
				indexs.insert(rand_int() % number);
			}

			Eigen::Matrix3d M;
			{
				Eigen::MatrixXd _srcMatrix(s, 3);
				Eigen::MatrixXd _dstMatrix(s, 3);
				std::set<int>::const_iterator iter = indexs.cbegin();
				for (int j = 0; j < s; j++, iter++)
				{
					_srcMatrix(j, 0) = srcPoints[*iter](0, 0);
					_srcMatrix(j, 1) = srcPoints[*iter](1, 0);
					_srcMatrix(j, 2) = 1.0;

					_dstMatrix(j, 0) = dstPoints[*iter](0, 0);
					_dstMatrix(j, 1) = dstPoints[*iter](1, 0);
					_dstMatrix(j, 2) = 1.0;
				}

				Eigen::VectorXd v = solveHomographyDLT(_srcMatrix, _dstMatrix);
				M << v(0), v(3), v(6),
					v(1), v(4), v(7),
					v(2), v(5), v(8);
				M.array() /= v(8);
			}

			// step 3: statistic
			std::vector<int> _inliners(number);
			{
				// TODO
				Eigen::MatrixXd _srcMatrix = srcMatrix * M;
				double _x, _y, _d2;
				double _threshold2 = threshold * threshold;
				int count = 0;
				for (int j = 0; j < number; j++)
				{
					_x = _srcMatrix(j, 0) / _srcMatrix(j, 2);
					_y = _srcMatrix(j, 1) / _srcMatrix(j, 2);
					_d2 = pow(_x - dstMatrix(j, 0), 2) + pow(_y - dstMatrix(j, 1), 2);
					if (_d2 <= _threshold2)
					{
						_inliners[j] = 1;
						count++;
					}
					else
					{
						_inliners[j] = 0;
					}
				}

				if (bestCount < count)
				{
					bestCount = count;
					inlinersMask.assign(_inliners.begin(), _inliners.end());
				}
			}
		}

		// step 4: solve
		Eigen::VectorXd v;
		Eigen::Matrix3d M;
		{
			Eigen::MatrixXd _srcMatrix(bestCount, 3);
			Eigen::MatrixXd _dstMatrix(bestCount, 3);

			int temp = 0;
			for (int j = 0; j < number; j++)
			{
				if (inlinersMask[j] == 0)
				{
					continue;
				}
				_srcMatrix(temp, 0) = srcPoints[j](0, 0);
				_srcMatrix(temp, 1) = srcPoints[j](1, 0);
				_srcMatrix(temp, 2) = 1.0;

				_dstMatrix(temp, 0) = dstPoints[j](0, 0);
				_dstMatrix(temp, 1) = dstPoints[j](1, 0);
				_dstMatrix(temp, 2) = 1.0;

				temp++;
			}

			v = solveHomographyDLT(_srcMatrix, _dstMatrix);
		}

		// step 5: optimization
		{
			ceres::Problem optimizationProblem;
			for (int i = 0; i < number; i++)
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

			std::cout << summary.BriefReport() << std::endl;
		}

		// step 6: calculate H
		M << v(0), v(3), v(6),
			v(1), v(4), v(7),
			v(2), v(5), v(8);
		M.array() /= v(8);
		H = dstT.inverse() * M * srcT;
		return true;
	}

	// for circleboard
	void ZZYCalibration::getObjectPoints1(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<Eigen::Vector3d>& objectPoints)
	{
		for (int row = 0; row < borderSize.width; ++row)
		{
			for (int col = 0; col < borderSize.height; ++col) 
			{
				objectPoints.push_back(Eigen::Vector3d(col * squareSize.height, row * squareSize.width, 0.0));
			}
		}
	}

	// for chessboard
	void ZZYCalibration::getObjectPoints2(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<Eigen::Vector3d>& objectPoints)
	{
		for (int row = 0; row < borderSize.height; ++row)
		{
			for (int col = 0; col < borderSize.width; ++col) 
			{
				objectPoints.push_back(Eigen::Vector3d(col * squareSize.width, row * squareSize.height, 0.0));
			}
		}
	}

	void ZZYCalibration::computeCameraCalibration(std::vector<std::vector<Eigen::Vector2d>>& imagePoints,
		std::vector<std::vector<Eigen::Vector3d>>& objectPoints,
		cv::Mat& cameraMatrix, cv::Mat& distCoeffs)throw(ZZYCalibrationFailure)
	{
		/*获取2D&3D之间的单应矩阵*/
		std::cout << "Start Find Homography" << std::endl;
		clock_t t1 = clock();
		const int number = imagePoints.size();
		std::vector<Eigen::Matrix3d> homographies;
		for (int i = 0; i < number; i++) {
			Eigen::Matrix3d H;
			std::vector<Eigen::Vector2d> objectPoints2d;
			for (auto& v : objectPoints[i]) {
				objectPoints2d.push_back(Eigen::Vector2d(v(0), v(1)));
			}
			bool ok = findHomographyByOpenCV(objectPoints2d, imagePoints[i], H);
			/*bool ok = findHomography(objectPoints2d, imagePoints[i], H, true);
			bool ok = findHomographyByRansac(objectPoints2d, imagePoints[i], H, true);*/
			homographies.push_back(H);
		}
		clock_t t2 = clock();
		std::cout << "End Find Homography: " << t2-t1 << "ms" << std::endl;
		/*计算初始相机内参*/
		std::cout << "Start Solve Init Camera Intrinstic" << std::endl;
		clock_t t3 = clock();
		Eigen::Matrix3d K = solveInitCameraIntrinstic(homographies);
		clock_t t4 = clock();
		std::cout << "End Solve Init Camera Intrinstic: " << t4 - t3 << "ms" << std::endl;
		/*计算初始相机外参*/
		std::cout << "Start Solve Init Camera Extrinstic" << std::endl;
		clock_t t5 = clock();
		std::vector<Eigen::Matrix3d> RList;
		std::vector<Eigen::Vector3d> tList;
		std::vector<Eigen::Vector3d> rList;
		solveInitCameraExtrinstic(homographies, K, RList, tList);
		for (auto& item : RList) {
			Eigen::Vector3d _r = RotationMatrix2Vector(item);
			rList.push_back(_r);
		}
		clock_t t6 = clock();
		std::cout << "End Solve Init Camera Extrinstic: " << t6 - t5 << "ms" << std::endl;
		/*初始化相机内参&畸变参数*/
		Eigen::Matrix3d KMatrix;
		Eigen::VectorXd distortion;
		/*优化相机内参&外参*/
		{
			/*K(0, 0) = 30000;
			K(1, 1) = 30000;
			K(0, 2) = 8000;
			K(1, 2) = 8000;*/
			ceres::Problem problem;
			double k5[9] = { K(0,0), K(1,1), K(0,2), K(1,2), 0., 0., 0., 0., 0. };
			double k4[8] = { K(0,0), K(1,1), K(0,2), K(1,2), 0., 0., 0., 0. };
			for (int i = 0; i < number; ++i) {
				for (int j = 0; j < imagePoints[i].size(); ++j) {
					if (mDistortionParaNum == 5){
						ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<ProjectCost, 2, 9, 3, 3>(
							new ProjectCost(objectPoints[i][j], imagePoints[i][j], 5));
						problem.AddResidualBlock(costFunction,
							nullptr,
							k5,
							rList[i].data(),
							tList[i].data()
						);
					}
					else if (mDistortionParaNum == 4) {
						ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<ProjectCost, 2, 8, 3, 3>(
							new ProjectCost(objectPoints[i][j], imagePoints[i][j], 4));
						problem.AddResidualBlock(costFunction,
							nullptr,
							k4,
							rList[i].data(),
							tList[i].data()
						);
					}
				}
			}
			std::cout << "Bundle Ajustment Solve Options:" << std::endl;
			ceres::Solver::Options options;
			options.minimizer_progress_to_stdout = false;
			options.linear_solver_type = ceres::DENSE_SCHUR;
			options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
			options.preconditioner_type = ceres::JACOBI;
			options.max_num_iterations = 120;
			options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
			options.function_tolerance = 1e-16;
			options.gradient_tolerance = 1e-10;
			ceres::Solver::Summary summary; 
			ceres::Solve(options, &problem, &summary);
 			if (!summary.IsSolutionUsable()){
				throw ZZYCalibrationFailure("Bundle Adjustment Failed ...");
			}
			else{
				std::cout << std::endl
					<< " Bundle Adjustment Statistics (Approximated RMSE):\n"
					<< " #views: " << number << "\n"
					<< " #residuals: " << summary.num_residuals << "\n"
					<< " #num_parameters: " << summary.num_parameters << "\n"
					<< " #num_parameter_blocks: " << summary.num_parameter_blocks << "\n"
					<< " #Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
					<< " #Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
					<< " #Time (s): " << summary.total_time_in_seconds << "\n"
					<< std::endl;
				/*获取畸变参数&内参*/
				Eigen::Matrix3d cameraMatrix_;
				Eigen::VectorXd distCoeffs_(5);
				if (mDistortionParaNum == 5)
				{
					cameraMatrix_ << k5[0], 0.0, k5[2], 0, k5[1], k5[3], 0, 0, 1;
					distCoeffs_ << k5[4], k5[5], k5[7], k5[8], k5[6];
				}
				else if (mDistortionParaNum == 4)
				{
					cameraMatrix_ << k4[0], 0.0, k4[2], 0, k4[1], k4[3], 0, 0, 1;
					distCoeffs_ << k4[4], k4[5], k4[6], k4[7], 0;
				}
				/*计算相机重投影误差*/
				std::vector<double> reprojectionErrors;
				double totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rList, tList, cameraMatrix_, distCoeffs_, reprojectionErrors);
				std::cout << " Average Reprojection Error: " << totalAvgErr << std::endl;
				for (size_t i = 0; i < reprojectionErrors.size(); i++){
					std::cout << " " << i << " projection error = " << reprojectionErrors[i] << std::endl;
				}
				/*相机内参&外参转化为cv格式*/
				KMatrix = cameraMatrix_;
				distortion = distCoeffs_;
				cv::eigen2cv(cameraMatrix_, cameraMatrix);
				cv::eigen2cv(distCoeffs_, distCoeffs);
			}
		}
		/*获取旋转矩阵*/
		for (int i = 0; i < rList.size(); i++) {
			Eigen::Matrix3d _R = RotationVector2Matrix(rList[i]);
			RList[i] = _R;
		}
		/*获取旋转矩阵序列*/
		for (int i = 0; i < RList.size(); i++){
			cv::Mat RMat; cv::eigen2cv(RList[i], RMat); mRListMat.push_back(RMat);
			cv::Mat tMat; cv::eigen2cv(tList[i], tMat); mtListMat.push_back(tMat);
		}
		/*获取特征点3D坐标*/
		std::vector<std::vector<Eigen::Vector3d>> vX3D;
		vX3D.resize(mImageNum);
		std::vector<double> vScale;
		for (int i = 0; i < mImageNum; i++){
			for (int j = 0; j < mImagePoints[i].size(); j++){
				Eigen::Vector3d x3D = Eigen::Vector3d::Zero();
				double scale = CalculateScale(mImagePoints[i][j], RList[i], tList[i], KMatrix, distortion, x3D);
				vX3D[i].push_back(x3D);
			}
		}
		this->mvX3D = vX3D;
	}

	void ZZYCalibration::DepthRecover(const Eigen::Matrix3d& K,
		const Eigen::VectorXd& distortion,
		const std::vector<Eigen::Matrix3d>& RList,
		const std::vector<Eigen::Vector3d>& tList,
		std::vector<std::vector<Eigen::Vector3d>>& vX3D)throw(ZZYCalibrationFailure)
	{
		vX3D.resize(mImageNum);

		for (int i = 0; i < mImageNum; i++)
		{

			const int indexi = i % mImageNum;
			const int indexj = (i + 1) % mImageNum;
			const int number1 = mImagePoints[indexi].size();
			const int number2 = mImagePoints[indexj].size();

			if (number1 != number2)
				throw ZZYCalibrationFailure("feature points error!");

			for (int j = 0; j < number1; j++)
			{
				Eigen::Vector2d kp1 = mImagePoints[indexi][j];
				Eigen::Vector2d kp2 = mImagePoints[indexj][j];

				Eigen::Vector2d kp1_un;
				Eigen::Vector2d kp2_un;

				UndistortionKeys(kp1, kp1_un, K, distortion);
				UndistortionKeys(kp2, kp2_un, K, distortion);

				Eigen::MatrixXd P1(3, 4);
				Eigen::MatrixXd P2(3, 4);

				P1.block(0, 0, 3, 3) = RList[indexi];
				P1.block(0, 3, 3, 1) = tList[indexi];
				P2.block(0, 0, 3, 3) = RList[indexj];
				P2.block(0, 3, 3, 1) = tList[indexj];

				P1 = K * P1;
				P2 = K * P2;

				Eigen::Vector3d x3D;
				Triangulate(kp1_un, kp2_un, P1, P2, x3D);

				vX3D[i].push_back(x3D);
			}
		}
	}

	void ZZYCalibration::Optimize(const Eigen::Matrix3d& K, std::vector<Eigen::Matrix3d>& RList, std::vector<Eigen::Vector3d>& tList, std::vector<std::vector<Eigen::Vector3d>>& vP3D)
	{
		ceres::Problem problem;

		const int number = vP3D[0].size();
		const int imgNum = mImageNum;

		std::vector<Eigen::Vector3d> rList;
		rList.resize(imgNum);

		for (int i = 0; i < imgNum; i++)
			rList[i] = RotationMatrix2Vector(RList[i]);

		std::vector<std::vector<pair<Eigen::Vector2d, Eigen::Vector2d>>> mvKeys;
		mvKeys.resize(imgNum);

		for (int i = 0; i < imgNum; i++)
		{
			for (int j = 0; j < number; j++)
			{
				const int indexi = i % imgNum;
				const int indexj = (i + 1) % imgNum;

				pair<Eigen::Vector2d, Eigen::Vector2d> key;
				key.first = mImagePoints[indexi][j];
				key.second = mImagePoints[indexj][j];

				mvKeys[i].push_back(key);
			}
		}

		for (int i = 0; i < imgNum; ++i)
		{
			for (int j = 0; j < number; ++j)
			{
				for (int k = 0; k < 2; k++)
				{
					if (k == 0)
					{
						ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<BundleAdjustment, 2, 3, 3, 3>(
							new BundleAdjustment(mvKeys[i][j].first[0], mvKeys[i][j].first[1], K));

						problem.AddResidualBlock(costFunction,
							nullptr,
							vP3D[i][j].data(),
							rList[i].data(),
							tList[i].data()
						);
					}
					else
					{
						ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<BundleAdjustment, 2, 3, 3, 3>(
							new BundleAdjustment(mvKeys[i][j].second[0], mvKeys[i][j].second[1], K));

						problem.AddResidualBlock(costFunction,
							nullptr,
							vP3D[i][j].data(),
							rList[(i + 1) % imgNum].data(),
							tList[(i + 1) % imgNum].data()
						);
					}
				}

			}
		}

		std::cout << "Bundle Ajustment Solve Options ..." << std::endl;

		ceres::Solver::Options options;
		options.minimizer_progress_to_stdout = false;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
		options.preconditioner_type = ceres::JACOBI;
		options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
		options.function_tolerance = 1e-16;
		options.gradient_tolerance = 1e-6;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		for (int i = 0; i < imgNum; i++)
			RList[i] = RotationVector2Matrix(rList[i]);
	}

	void ZZYCalibration::UndistortionKeys(const Eigen::Vector2d& vUnKeys, Eigen::Vector2d& vKeys, const Eigen::Matrix3d& K, const Eigen::VectorXd& distortion)
	{
		const double fx = K(0, 0);
		const double fy = K(1, 1);
		const double cx = K(0, 2);
		const double cy = K(1, 2);

		const double k1 = distortion(0);
		const double k2 = distortion(1);
		const double p1 = distortion(2);
		const double p2 = distortion(3);
		const double k3 = distortion(4);

		double u = vUnKeys(0);
		double v = vUnKeys(1);

		double xp = (u - cx) / fx;
		double yp = (v - cy) / fy;

		double r_2 = xp * xp + yp * yp;

		double xdis = xp * (double(1.) + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + double(2.) * p1 * xp * yp + p2 * (r_2 + double(2.) * xp * xp);
		double ydis = yp * (double(1.) + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2) + p1 * (r_2 + double(2.) * yp * yp) + double(2.) * p2 * xp * yp;

		double u_un = fx * xdis + cx;
		double v_un = fy * ydis + cy;

		vKeys = Eigen::Vector2d(u_un, v_un);

	}


	// sU = K*[R|t]*P
	double ZZYCalibration::CalculateScale(const Eigen::Vector2d& imagePoint,
		const Eigen::Matrix3d& R,
		const Eigen::Vector3d& t,
		const Eigen::Matrix3d& K,
		const Eigen::VectorXd& distortion,
		Eigen::Vector3d& x3D)
	{
		double Zw = 0.0;
		Eigen::Vector2d unImagePoint;
		UndistortionKeys(imagePoint, unImagePoint, K, distortion);

		Eigen::Vector3d leftSideMatrix = Eigen::Vector3d::Zero();
		Eigen::Vector3d rightSideMatrix = Eigen::Vector3d::Zero();

		Eigen::Vector3d imageHomo = Eigen::Vector3d::Ones();
		imageHomo(0) = unImagePoint(0);
		imageHomo(1) = unImagePoint(1);

		leftSideMatrix = R.inverse() * K.inverse() * imageHomo;
		rightSideMatrix = R.inverse() * t;

		double scale = Zw + rightSideMatrix(2, 0) / leftSideMatrix(2, 0);

		//x3D = R.inverse() * (scale * K.inverse() * imageHomo - t);
		// camera coordinate
		x3D = scale * K.inverse() * imageHomo;

		return scale;
	}

	void ZZYCalibration::Triangulate(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2, const Eigen::MatrixXd& P1, const Eigen::MatrixXd& P2, Eigen::Vector3d& x3D)
	{
		Eigen::MatrixXd A(4, 4);

		A.row(0) = kp1(0) * P1.row(2) - P1.row(0);
		A.row(1) = kp1(1) * P1.row(2) - P1.row(1);
		A.row(2) = kp2(0) * P2.row(2) - P2.row(0);
		A.row(3) = kp2(1) * P2.row(2) - P2.row(1);

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinU | ComputeThinV);
		Eigen::MatrixXd V = svd.matrixV();
		Eigen::VectorXd v = V.rightCols(1);

		x3D(0) = v(0) / v(3);
		x3D(1) = v(1) / v(3);
		x3D(2) = v(2) / v(3);
	}

	Eigen::VectorXd ZZYCalibration::constructVector(const Matrix3d& H, int i, int j)
	{
		i -= 1;
		j -= 1;

		Eigen::VectorXd v(6);
		//TODO
		v << H(0, i) * H(0, j), H(0, i)* H(1, j) + H(1, i) * H(0, j), H(1, i)* H(1, j), H(2, i)* H(0, j) + H(0, i) * H(2, j), H(2, i)* H(1, j) + H(1, i) * H(2, j), H(2, i)* H(2, j);
		return v;
	}

	Eigen::Matrix3d ZZYCalibration::solveInitCameraIntrinstic(const std::vector<Eigen::Matrix3d>& homographies)
	{
		const int number = homographies.size();
		Eigen::MatrixXd  A(2 * number, 6);
		for (int i = 0; i < number; i++)
		{
			Eigen::VectorXd v1 = constructVector(homographies[i], 1, 2);
			Eigen::VectorXd v11 = constructVector(homographies[i], 1, 1);
			Eigen::VectorXd v22 = constructVector(homographies[i], 2, 2);
			Eigen::VectorXd v2 = v11 - v22;

			for (int j = 0; j < 6; j++)
			{
				A(2 * i, j) = v1(j);
				A(2 * i + 1, j) = v2(j);
			}
		}

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinV);
		Eigen::MatrixXd V = svd.matrixV();
		Eigen::MatrixXd b = V.rightCols(1);

		/*std::cout << "b = " << b << std::endl;*/

		double B11 = b(0), B12 = b(1), B22 = b(2), B13 = b(3), B23 = b(4), B33 = b(5);

		if (B11 < 0 || B22 < 0 || B33 < 0) std::cerr << "Error Occurred!" << std::endl;
		if (B11 < 0) B11 = -B11; if (B22 < 0) B22 = -B22; if (B33 < 0) B33 = -B33;

		double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
		double s = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
		double fx = std::sqrt(s / B11);
		double fy = std::sqrt(s * B11 / (B11 * B22 - B12 * B12));
		double c = -B12 * fx * fx * fy / s;
		double u0 = c * v0 / fx - B13 * fx * fx / s;

		Eigen::Matrix3d K;
		K << fx, c, u0,
			0, fy, v0,
			0, 0, 1;
		
		return K;
	}

	void ZZYCalibration::solveInitCameraExtrinstic(const std::vector<Matrix3d>& homographies, const Eigen::Matrix3d& K, 
		std::vector<Eigen::Matrix3d>& RList, std::vector<Vector3d>& tList)
	{
		const int number = homographies.size();
		Eigen::Matrix3d K_inv = K.inverse();
		for (int i = 0; i < number; i++)
		{
			Eigen::Vector3d r0, r1, r2;
			r0 = K_inv * homographies[i].col(0);
			r1 = K_inv * homographies[i].col(1);
			
			double s0 = ceres::sqrt(r0.dot(r0));
			double s1 = ceres::sqrt(r1.dot(r1));

			r0.array().col(0) /= s0;
			r1.array().col(0) /= s1;
			r2 = r0.cross(r1);

			Eigen::Vector3d t = K_inv * homographies[i].col(2) / s0;
			
			Eigen::Matrix3d R;
			R.array().col(0) = r0;
			R.array().col(1) = r1;
			R.array().col(2) = r2;
			
			/*std::cout << "R: " << R << std::endl;
			std::cout << "t: " << t.transpose() << std::endl;*/
			
			RList.push_back(R);
			tList.push_back(t);
		}
	}

	/**
	* breief: compute reprojection error
	* @param objectPoints
	* @param imagePoints
	* @param rvecs
	* @param tvecs
	* @param cameraMatrix
	* @param distCoeffs
	* @param perViewErrors
	* @return
	*/
	double ZZYCalibration::computeReprojectionErrors(const vector<vector<Eigen::Vector3d>>& objectPoints, const vector<vector<Eigen::Vector2d>>& imagePoints, const vector<Eigen::Vector3d>& rvecs,
		const vector<Eigen::Vector3d>& tvecs, const Eigen::Matrix3d& cameraMatrix, const Eigen::VectorXd& distCoeffs, vector<double>& perViewErrors)throw(ZZYCalibrationFailure)
	{
		int totalPoints = 0;
		double totalErr = 0.0;
		perViewErrors.resize(objectPoints.size());
		const int number = objectPoints.size();

		double k1 = distCoeffs(0), k2 = distCoeffs(1), k3 = distCoeffs(4);
		double p1 = distCoeffs(2), p2 = distCoeffs(3);
		double fx = cameraMatrix(0, 0), fy = cameraMatrix(1, 1);
		double cx = cameraMatrix(0, 2), cy = cameraMatrix(1, 2);
		
		for (int i = 0; i < number; i++)
		{
			Eigen::Matrix3d R = RotationVector2Matrix(rvecs[i]);
			const int n = objectPoints[i].size();
			double _errSum = 0.0;
			for (int j = 0; j < n; j++)
			{
				Eigen::Vector3d cam = R * objectPoints[i][j] + tvecs[i];
				double xp = cam(0) / cam(2);
				double yp = cam(1) / cam(2);
				
				double r2 = xp * xp + yp * yp;
				double xdis = xp * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp);
				double ydis = yp * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) + p1 * (r2 + 2 * yp * yp) + 2 * p2 * xp * yp;
				double u = fx * xdis + cx;
				double v = fy * ydis + cy;

				double _err = std::sqrt(pow(u - imagePoints[i][j](0), 2) + pow(v - imagePoints[i][j](1), 2));
				_errSum += _err;
			}
			perViewErrors[i] = _errSum / n;
			totalErr += _errSum;
			totalPoints += n;
		}
		return totalErr / totalPoints;
	}
}