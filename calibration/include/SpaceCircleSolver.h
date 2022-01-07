#pragma once
#include<Eigen/Core>
#include<Eigen/SVD>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/calib3d.hpp>
#include<iostream>
#include"Random.h"

using namespace std;
using namespace Eigen;

namespace MtSpaceCircleSolver
{
	typedef long long int lli;
	
	class SpaceCircleSolverFailure
	{
	public:
		SpaceCircleSolverFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class SpaceCircleSolver
	{
	public:
		SpaceCircleSolver(const std::vector<Eigen::Vector3d>& data, const float& threshold, const bool& isRansac) :
			mData(data),mThreshold(threshold), mbRansac(isRansac) {};
		void compute(Eigen::VectorXd& circlePara,double& radius);
	private:
		
		float mThreshold;
		const bool mbRansac;
		std::vector<Eigen::Vector3d> mData;
		std::vector<std::vector<int>> mvSets;
		
		// method
		lli combination(const lli& n, const lli& a);
		void calculatePlanarPara(const std::vector<Eigen::Vector3d>& data, Eigen::VectorXd& planarPara)throw(SpaceCircleSolverFailure);
		void calculateCirclePara(const std::vector<Eigen::Vector3d>& data, Eigen::VectorXd& center, double& radius)throw(SpaceCircleSolverFailure);
		
		float calculateDistance(const Eigen::Vector3d& p1, const Eigen::VectorXd& p2);
		float calculateRadius(const std::vector<Eigen::Vector3d>& data, Eigen::VectorXd& center);
		void calculateCircleParaByRansac(const float& p, const float& w, const int& s, Eigen::VectorXd& center, double& radius);
		int checkCirclePara(const float& threshold, const std::vector<Eigen::Vector3d>& data, std::vector<bool>& vbInliers, Eigen::VectorXd& center);
	};
}