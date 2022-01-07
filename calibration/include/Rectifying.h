#pragma once

#include<Eigen/Core>
#include<Eigen/SVD>
#include<Eigen/Dense>

//Reference: Computing Rectifying Homographies for Stereo Vision

class Rectifying
{
public:
	Rectifying();
	~Rectifying() {};
private:
	int				mImageWidthLeft;
	int				mImageHeightLeft;
	int				mImageWidthRight;
	int				mImageHeightRight;
	
	Eigen::Matrix3d mFundamental;

	Eigen::Vector3d mLeftEpipole;
	Eigen::Vector3d mRightEpipole;

	Eigen::Matrix3d mLeftProjectionTr;
	Eigen::Matrix3d mRightProjectionTr;

	Eigen::Matrix3d mLeftSimilarityTr;
	Eigen::Matrix3d mRightSimilarityTr;

	Eigen::Matrix3d mLeftShearingTr;
	Eigen::Matrix3d mRightShearingTr;

	bool CalculateProjectionTransform();

private:
	
	Eigen::Matrix3d skew(const Eigen::Vector3d& vec);
	Eigen::Vector2d Optimize(const Eigen::Matrix2d& A, const Eigen::Matrix2d& B);
};