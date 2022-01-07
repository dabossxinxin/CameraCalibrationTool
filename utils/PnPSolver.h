#pragma once
#include <Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>

namespace MtPnPSolver
{
	class PnPSolver
	{
	public:
		PnPSolver(const std::vector<Eigen::Vector3d>& pts3d, std::vector<Eigen::Vector2d>& pts2d, const std::string& file)
			:mPts3d(pts3d), mPts2d(pts2d), mStrSettingFile(file) {};

		void SolvePnP(Eigen::Matrix3d& R, Eigen::Vector3d& t);
	private:
		std::string mStrSettingFile;
		std::vector<Eigen::Vector3d> mPts3d;
		std::vector<Eigen::Vector2d> mPts2d;

		/************** Functions for DLT solver ******************/
		bool solvePnPbyDLT(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& pts3d, const std::vector<Eigen::Vector2d>& pts2d, Eigen::Matrix3d& R, Eigen::Vector3d& t);

		/************** Functions for P3P solver ******************/
		int solveForLengths(double lengths[4][3], const double distances[3], const double cosines[3]);
		bool solvePnPbyP3P(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& pts3d, const std::vector<Eigen::Vector2d>& pts2d, Eigen::Matrix3d& R, Eigen::Vector3d& t);

		/************** Functions for EPnP solver ******************/
		bool solvePnPbyEPnP(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& pts3d, const std::vector<Eigen::Vector2d>& pts2d, Eigen::Matrix3d& R, Eigen::Vector3d& t);
		void selectControlPoints(const std::vector<Eigen::Vector3d>& pts3d, std::vector<Eigen::Vector3d>& control_points);
		void computeHomogeneousBarycentricCoordinates(const std::vector<Eigen::Vector3d>& pts3d, const std::vector<Eigen::Vector3d>& control_points, std::vector<Eigen::Vector4d>& hb_coordinates);
		void constructM(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector4d>& hb_coordinates, const std::vector<Eigen::Vector2d>& pts2d, Eigen::MatrixXd& M);
		void getFourEigenVectors(const Eigen::MatrixXd& M, Eigen::Matrix<double, 12, 4>& eigen_vectors);
		void computeL(const Eigen::Matrix<double, 12, 4>& eigen_vectors, Eigen::Matrix<double, 6, 10>& L);
		void computeRho(const std::vector<Eigen::Vector3d>& control_points, Eigen::Matrix<double, 6, 1>& rho);
		void solveBetaN2(const Eigen::Matrix<double, 12, 4>& eigen_vectors, const Eigen::Matrix<double, 6, 10>& L, const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d& betas);
		void solveBetaN3(const Eigen::Matrix<double, 12, 4>& eigen_vectors, const Eigen::Matrix<double, 6, 10>& L, const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d& betas);
		void solveBetaN4(const Eigen::Matrix<double, 12, 4>& eigen_vectors, const Eigen::Matrix<double, 6, 10>& L, const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d& betas);
		void optimizeBeta(const Eigen::Matrix<double, 6, 10>& L, const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d& betas);
		void computeCameraControlPoints(const Eigen::Matrix<double, 12, 4>& eigen_vectors, const Eigen::Vector4d& betas, std::vector<Eigen::Vector3d>& camera_control_points);
		bool isGoodBetas(const std::vector<Eigen::Vector3d>& camera_control_points);
		void rebuiltPts3dCamera(const std::vector<Eigen::Vector3d>& camera_control_points, const std::vector<Eigen::Vector4d>& hb_coordinates, std::vector<Eigen::Vector3d>& pts3d_camera);
		void computeRt(const std::vector<Eigen::Vector3d>& pts3d_camera, const std::vector<Eigen::Vector3d>& pts3d_world, Eigen::Matrix3d& R, Eigen::Vector3d& t);
		double reprojectionError(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& pts3d_world, const std::vector<Eigen::Vector2d>& pts2d, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	};
}
