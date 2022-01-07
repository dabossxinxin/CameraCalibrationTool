#pragma once
#include "SpaceLineSolver.h"

namespace MtSpaceLineSolver
{
	SpaceLineSolverFailure::SpaceLineSolverFailure(const char* msg) :Message(msg) {};
	const char* SpaceLineSolverFailure::GetMessage() { return Message; }

	void SpaceLineSolver::compute(Eigen::Vector3d& normal)
	{
		Eigen::VectorXd para;
		calculateParameterWithLS(mData, para);

		const double a = para(0);
		const double b = para(1);
		const double c = para(2);
		const double d = para(3);

		normal(2) = 1.0;
		normal(1) = c;
		normal(0) = a;
		
		normal.normalize();
	}

	void SpaceLineSolver::calculateParameterWithLS(const std::vector<Eigen::Vector3d>& data, Eigen::VectorXd& para)
	{
		const int number = data.size();
		
		Eigen::VectorXd A(2);
		Eigen::VectorXd B(2);
		Eigen::MatrixXd M(2, number);
		Eigen::VectorXd X(number);
		Eigen::VectorXd Y(number);

		for (int i = 0; i < number; i++)
		{
			M(0, i) = data[i](2);
			M(1, i) = 1.0;

			X(i) = data[i](0);
			Y(i) = data[i](1);
		}

		Eigen::MatrixXd Mt = M.transpose();
		Eigen::MatrixXd MtMinv = (M * Mt).inverse();
		
		A = MtMinv * M * X;
		B = MtMinv * M * Y;
		
		Eigen::VectorXd result(4);
		
		result(0) = A(0);
		result(1) = A(1);
		result(2) = B(0);
		result(3) = B(1);

		para = result;
		mCoef = result;
	}

	void SpaceLineSolver::RMS()
	{
		const int number = mData.size();
		const double a = mCoef(0);
		const double b = mCoef(1);
		const double c = mCoef(2);
		const double d = mCoef(3);
		
		double err = 0.0;
		for (int i = 0; i < number; i++)
		{
			double dx = a * mData[i](2) + b - mData[i](0);
			double dy = c * mData[i](2) + d - mData[i](1);
			err += sqrt(dx * dx + dy * dy);
		}
		
		std::cout << "Space Line Fit RMS: " << err / (number*10) << std::endl;
	}
}