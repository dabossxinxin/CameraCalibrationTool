#include "Rectifying.h"
#include <Eigen/Cholesky>

Rectifying::Rectifying()
{
	mLeftProjectionTr = Eigen::Matrix3d::Identity(3, 3);
	mRightProjectionTr = Eigen::Matrix3d::Identity(3, 3);
	mLeftSimilarityTr = Eigen::Matrix3d::Identity(3, 3);
	mRightSimilarityTr = Eigen::Matrix3d::Identity(3, 3);
	mLeftShearingTr = Eigen::Matrix3d::Identity(3, 3);
	mRightShearingTr = Eigen::Matrix3d::Identity(3, 3);
}

bool Rectifying::CalculateProjectionTransform()
{
	// 函数计算必要的参数
	const int widthL = this->mImageWidthLeft;
	const int heightL = this->mImageHeightLeft;
	const int widthR = this->mImageWidthRight;
	const int heightR = this->mImageHeightRight;
	const Eigen::Vector3d epipoleL = this->mLeftEpipole;
	const Eigen::Vector3d epipoleR = this->mRightEpipole;
	const Eigen::Matrix3d epipoleLMat = skew(epipoleL);
	const Eigen::Matrix3d epipoleRMat = skew(epipoleR);
	const Eigen::Matrix3d fundamental = this->mFundamental;

	// 计算PPT
	Eigen::Matrix3d ppLeft = Eigen::Matrix3d::Zero(3, 3);
	Eigen::Matrix3d ppRight = Eigen::Matrix3d::Zero(3, 3);
	const double multiL = widthL*heightL / 12.0;
	ppLeft(0, 0) = multiL*(widthL*widthL - 1);
	ppLeft(1, 1) = multiL*(heightL*heightL - 1);
	const double multiR = widthR*heightR / 12.0;
	ppRight(0, 0) = multiR*(widthR*widthR - 1);
	ppRight(1, 1) = multiR*(heightR*heightR - 1);
	
	// 计算PcPcT
	Eigen::Matrix3d pcpcLeft = Eigen::Matrix3d::Zero(3, 3);
	Eigen::Matrix3d pcpcRight = Eigen::Matrix3d::Zero(3, 3);
	pcpcLeft(0, 0) = 0.25*(widthL - 1)*(widthL - 1);
	pcpcLeft(0, 1) = 0.25*(widthL - 1)*(heightL - 1);
	pcpcLeft(0, 2) = 0.5*(widthL - 1);
	pcpcLeft(1, 0) = pcpcLeft(0, 1);
	pcpcLeft(1, 1) = 0.25*(heightL - 1)*(heightL - 1);
	pcpcLeft(1, 2) = 0.5*(heightL - 1);
	pcpcLeft(2, 0) = pcpcLeft(0, 2);
	pcpcLeft(2, 1) = pcpcLeft(1, 2);
	pcpcLeft(2, 2) = 1.0;

	pcpcRight(0, 0) = 0.25*(widthR - 1)*(widthR - 1);
	pcpcRight(0, 1) = 0.25*(widthR - 1)*(heightR - 1);
	pcpcRight(0, 2) = 0.5*(widthR - 1);
	pcpcRight(1, 0) = pcpcRight(0, 1);
	pcpcRight(1, 1) = 0.25*(heightR - 1)*(heightR - 1);
	pcpcRight(1, 2) = 0.5*(heightR - 1);
	pcpcRight(2, 0) = pcpcRight(0, 2);
	pcpcRight(2, 1) = pcpcRight(1, 2);
	pcpcRight(2, 2) = 1.0;

	// 计算做相机与右相机的A&B矩阵
	Eigen::Matrix2d ALeft, BLeft;
	Eigen::Matrix2d ARight, BRight;
	ALeft = (epipoleLMat.transpose()*ppLeft*epipoleLMat).block(0,0,2,2);
	BLeft = (epipoleLMat.transpose()*pcpcLeft*epipoleLMat).block(0,0,2,2);
	ARight = (fundamental.transpose()*ppRight*fundamental).block(0,0,2,2);
	BRight = (fundamental.transpose()*pcpcRight*fundamental).block(0,0,2,2);

	Eigen::Vector2d zInitLeft = Optimize(ALeft, BLeft);
	Eigen::Vector2d zInitRight = Optimize(ARight, BRight);
	Eigen::Vector2d zInit = zInitLeft + zInitRight; zInit.normalize();

	// 使用LM优化z的值
	return true;
}

Eigen::Matrix3d Rectifying::skew(const Eigen::Vector3d& vec)
{
	Eigen::Matrix3d matrix;
	matrix(0, 0) = 0.; matrix(0, 1) = -vec(2); matrix(0, 2) = vec(1);
	matrix(1, 0) = vec(2); matrix(1, 1) = 0.; matrix(1, 2) = -vec(0);
	matrix(2, 0) = -vec(1); matrix(2, 1) = vec(0); matrix(2, 2) = 0.;
	return matrix;
}

Eigen::Vector2d Rectifying::Optimize(const Eigen::Matrix2d& A, const Eigen::Matrix2d& B)
{
	const Eigen::Matrix2d D = A.llt().matrixL();
	const Eigen::Matrix2d Dinv = D.inverse();
	const Eigen::Matrix2d Dt = D.transpose();
	const Eigen::Matrix2d Dtinv = Dt.inverse();
	const Eigen::Matrix2d DtinvBDinv = Dtinv*B*Dinv;

	Eigen::EigenSolver<Eigen::Matrix2d> es(DtinvBDinv);
	Eigen::MatrixXcd evecs = es.eigenvectors();
	Eigen::MatrixXcd evals = es.eigenvalues();
	Eigen::MatrixXd evalsReal;
	evalsReal = evals.real();
	Eigen::MatrixXf::Index evalsMax;
	evalsReal.rowwise().sum().maxCoeff(&evalsMax);
	Eigen::Vector2d result;
	result << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
	return result;
}