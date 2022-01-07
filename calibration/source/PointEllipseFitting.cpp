#include"PointEllipseFitting.h"

namespace MtEllipseFitting
{
	EllipseSolverFailure::EllipseSolverFailure(const char* msg) :Message(msg) {};
	const char* EllipseSolverFailure::GetMessage() { return Message; }

	void EllipseSolver::compute()throw(EllipseSolverFailure)
	{
		cv::Mat1d D1;
		cv::Mat1d D2;
		constructD(D1, D2, mbGravity);
		
		cv::Mat1d S1 = D1.t() * D1;
		cv::Mat1d S2 = D1.t() * D2;
		cv::Mat1d S3 = D2.t() * D2;

		cv::Mat1d S3_inv = cv::Mat::zeros(3,3,CV_64F);
		cv::invert(S3, S3_inv, cv::DECOMP_SVD);
		cv::Mat1d T = -S3_inv * S2.t();
		cv::Mat1d M = S1 + S2 * T;

		cv::Mat1d C_inv = cv::Mat::zeros(3, 3, CV_64F);
		C_inv(0, 0) = 0; C_inv(0, 1) = 0; C_inv(0, 2) = 0.5;
		C_inv(1, 0) = 0; C_inv(1, 1) = -1; C_inv(1, 2) = 0;
		C_inv(2, 0) = 0.5; C_inv(2, 1) = 0; C_inv(2, 2) = 0;
		
		M = C_inv * M;

		cv::SVD svd(M, cv::SVD::FULL_UV);
		cv::Mat1d U = svd.u;
		cv::Mat1d S = svd.w;
		cv::Mat1d Vt = svd.vt;

		/*std::cout << "S:" << std::endl << S << std::endl;
		std::cout << "U:" << std::endl << U << std::endl;
		std::cout << "Vt:" << std::endl << Vt << std::endl;*/

		// get smallest eignvalue
		int index;
		GetSmallestEigenVal(S, index);
		
		cv::Mat1d a1 = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat1d a2 = cv::Mat::zeros(3, 1, CV_64F);
		
		a1(0, 0) = Vt(index, 0);
		a1(1, 0) = Vt(index, 1);
		a1(2, 0) = Vt(index, 2);

		a2 = T * a1;

		cv::Mat1d conic = cv::Mat::zeros(6, 1, CV_64F);
		conic(0, 0) = a1(0, 0);
		conic(1, 0) = a1(1, 0);
		conic(2, 0) = a1(2, 0);
		conic(3, 0) = a2(0, 0);
		conic(4, 0) = a2(1, 0);
		conic(5, 0) = a2(2, 0);

		// convert conic parameter to ellipse parameter
		mEllipsePara = convertConicToParametric(conic);
	}

	cv::Mat1d EllipseSolver::convertConicToParametric(cv::Mat1d& par)
	{
		// theta = arctan(B/(A-C))/2;
		// Au = D*cos(theta)+E*sin(theta)
		// Av = -D*cos(theta)+E*sin(theta)
		// Auu = A*cos(theta)^2+C*sin(theta)^2+B*sin(theta)*cos(theta)
		// Avv = A*sin(theta)^2+C*cos(theta)^2-B*sin(theta)*cos(theta)
		// tuCenter = -Au/(2*Auu)
		// tvCenter = -Av/(2*Avv)
		// wCenter = F - Auu*tuCenter^2 - Avv*tvCenter^2
		// uCenter = tuCenter*cos(theta) - tvCenter*sin(theta)
		// vCenter = tuCenter*sin(theta) + tvCenter*cos(theta)
		// Ru = sign(-wCenter/Auu)*sqrt(abs(-wCenter/Auu))
		// Rv = sign(-wCenter/Avv)*sqrt(abs(-wCenter/Avv))
		cv::Mat1d ell = cv::Mat::zeros(5, 1, CV_64F);

		double thetarad = 0.5 * atan2(par(1, 0), par(0, 0) - par(2, 0));
		double cost = cos(thetarad);
		double sint = sin(thetarad);
		double sin_squared = sint * sint;
		double cos_squared = cost * cost;
		double cos_sin = sint * cost;

		double Ao = par(5, 0);
		double Au = par(3, 0) * cost + par(4, 0) * sint;
		double Av = -par(3, 0) * sint + par(4, 0) * cost;
		double Auu = par(0, 0) * cos_squared + par(2, 0) * sin_squared + par(1, 0) * cos_sin;
		double Avv = par(0, 0) * sin_squared + par(2, 0) * cos_squared - par(1, 0) * cos_sin;

		double tuCentre = -Au / (2 * Auu);
		double tvCentre = -Av / (2. * Avv);
		double wCentre = Ao - Auu * tuCentre * tuCentre - Avv * tvCentre * tvCentre;

		double uCentre = tuCentre * cost - tvCentre * sint;
		double vCentre = tuCentre * sint + tvCentre * cost;

		double Ru = -wCentre / Auu;
		double Rv = -wCentre / Avv;

		Ru = sqrt(abs(Ru)) * sign(Ru);
		Rv = sqrt(abs(Rv)) * sign(Rv);

		double centrex = uCentre;
		double centrey = vCentre;
		double axea = Ru;
		double axeb = Rv;
		double angle = -(thetarad/PI)*180.0;

		if (mbGravity)
		{
			ell.at<double>(1, 0) = centrex + mGravityX;
			ell.at<double>(0, 0) = centrey + mGravityY;
		}
		else
		{
			ell.at<double>(1, 0) = centrex;
			ell.at<double>(0, 0) = centrey;
		}
		
		ell.at<double>(3, 0) = axea;
		ell.at<double>(2, 0) = axeb;
		ell.at<double>(4, 0) = angle;

		/*std::cout << "centerx: " << centrex << endl
			<< "centery: " << centrey << endl
			<< "axea: " << axea << endl
			<< "axeb: " << axeb << endl
			<< "angle: " << angle << endl;*/

		return ell;
	}

	void EllipseSolver::GetSmallestEigenVal(const cv::Mat1d& S, int& index)
	{
		double Min;
		const int number = S.rows;
		for (int it = 0; it < number; it++)
		{
			if (S(it, 0) > 0)
			{
				Min = S(it, 0);
				index = it;
			}
		}

		for (int it = 0; it < number; it++)
		{
			if (S(it, 0) > 0 && S(it, 0) < Min)
			{
				Min = S(it, 0);
				index = it;
			}
		}
	}

	void EllipseSolver::Gravity(const std::vector<float>& input, std::vector<float>& output, const string& direction)
	{
		double gravity = 0.0;
		const int number = input.size();
		const double N_inv = 1.0 / number;
		for (int it = 0; it < number; it++)
		{
			gravity += input[it];
		}
		gravity *= N_inv;
		
		if (direction == "x")
		{
			mGravityX = gravity;
		}
		else if(direction == "y")
		{
			mGravityY = gravity;
		}

		output.resize(number);
		for (int it = 0; it < number; it++)
		{
			output[it] = input[it] - gravity;
		}
	}

	void EllipseSolver::constructD(cv::Mat1d& D1, cv::Mat1d& D2, const bool& gravity)throw(EllipseSolverFailure)
	{
		const int numberx = mX.size();
		const int numbery = mY.size();
		
		if (numberx != numbery || numberx == 0 || numbery == 0)
		{
			throw EllipseSolverFailure("Input Error!");
			return;
		}

		std::vector<float> GX;
		std::vector<float> GY;

		if (gravity)
		{
			Gravity(mX, GX, "x");
			Gravity(mY, GY, "y");
		}
		else
		{
			GX = mX;
			GY = mY;
		}

		const int number = numberx;
		cv::Mat1d D1t = cv::Mat::zeros(number, 3, CV_64F);
		cv::Mat1d D2t = cv::Mat::zeros(number, 3, CV_64F);

 		for (int it = 0; it < number; it++)
		{
			D1t(it, 0) = GX[it] * GX[it];
			D1t(it, 1) = GX[it] * GY[it];
			D1t(it, 2) = GY[it] * GY[it];
			
			D2t(it, 0) = GX[it];
			D2t(it, 1) = GY[it];
			D2t(it, 2) = 1;
		}

		D1 = D1t;
		D2 = D2t;
	}
}

