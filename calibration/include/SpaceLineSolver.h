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

namespace MtSpaceLineSolver
{
	typedef long long int lli;
	
	class SpaceLineSolverFailure
	{
	public:
		SpaceLineSolverFailure(const char* msg);
		const char* GetMessage();
	private:
		const char* Message;
	};

	class SpaceLineSolver
	{
	public:
		SpaceLineSolver(const std::vector<Eigen::Vector3d>& data, const float& threshold, const bool& isRansac) :
			mData(data),mThreshold(threshold), mbRansac(isRansac) {};
		void compute(Eigen::Vector3d& normal);
		void RMS();
	private:
		
		float mThreshold;
		const bool mbRansac;
		std::vector<Eigen::Vector3d> mData;
		std::vector<std::vector<int>> mvSets;
		Eigen::Vector4d mCoef;
		
		// method
		lli combination(const lli& n, const lli& a);
		void calculateParameterWithLS(const std::vector<Eigen::Vector3d>& data, Eigen::VectorXd& para);
	};
}