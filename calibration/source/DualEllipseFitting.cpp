#include "DualEllipseFitting.h"

namespace MtDualEllipseFitting
{
	int EllipseSolver::DualConicFitting(const std::vector<cv::Point_<double>>& roi_pixel, const cv::Mat& dx, const cv::Mat& dy, cv::Mat& conic, const bool& Normalization)
	{
		/*Ref : Precise ellipse estimation without contour point extraction,
		Jean-Nicolas Ouellet & Patrick Hebert, Machine Vision and Applications(2009) 21:59-67
		2009-Ouellet-Hebert-Precise ellipse estimation without contour point extraction.pdf*/

		int error = NO_ERROR;
		int roi_size = roi_pixel.size();

		//calculate the parameter of line
		//a = dx at the contour points 
		//b = dy at the contour points
		//c = -(dx*x + dy*y)
		cv::Mat1d a = cv::Mat::zeros(roi_size, 1, CV_64F);
		cv::Mat1d b = cv::Mat::zeros(roi_size, 1, CV_64F);
		cv::Mat1d c = cv::Mat::zeros(roi_size, 1, CV_64F);

		for (int i = 0; i < roi_size; i++)
		{
			a(i, 0) = dx.at<double>(roi_pixel[i]);
			b(i, 0) = dy.at<double>(roi_pixel[i]);
			c(i, 0) = -(a(i,0)*roi_pixel[i].x+b(i,0)*roi_pixel[i].y);
		}

		cv::Mat1d H = cv::Mat::zeros(3, 3, CV_64F);

		if (Normalization)
		{
			cv::Mat1d M = cv::Mat::zeros(roi_size, 2, CV_64F);
			cv::Mat1d B = -c;
			for (int i = 0; i < roi_size; i++)
			{
				M(i, 0) = -b(i, 0);
				M(i, 1) = a(i, 0);
			}
			// finding the pseudo inverse of the non-square matrix
			// pinv(A) = A' * (A * A')^-1
			invert(M, M, cv::DECOMP_SVD);
			//Mat1d mpts = (M.t()*(M*M.t()).inv()) * B;
			cv::Mat1d mpts = M * B;

			H(0, 0) = 1; H(0, 2) = mpts(0, 0);
			H(1, 1) = 1; H(1, 2) = mpts(1, 0);
			H(2, 2) = 1;

			cv::Mat1d Lnorm = cv::Mat::zeros(roi_size, 3, CV_64F);
			for (int i = 0; i < roi_size; i++)
			{
				Lnorm(i, 0) = a(i, 0);
				Lnorm(i, 1) = b(i, 0);
				Lnorm(i, 2) = c(i, 0);
			}
			Lnorm = (H.t() * Lnorm.t()).t();

			for (int i = 0; i < roi_size; i++)
			{
				a(i, 0) = Lnorm(i, 0);
				b(i, 0) = Lnorm(i, 1);
				c(i, 0) = Lnorm(i, 2);
			}
			mpts.release();
			B.release();
			M.release();
			Lnorm.release();
		}

		std::vector<double> a2, ab, b2, ac, bc, c2;

		for (int i = 0; i < roi_size; i++)
		{
			a2.push_back(a(i, 0) * a(i, 0));
			ab.push_back(a(i, 0) * b(i, 0));
			b2.push_back(b(i, 0) * b(i, 0));
			ac.push_back(a(i, 0) * c(i, 0));
			bc.push_back(b(i, 0) * c(i, 0));
			c2.push_back(c(i, 0) * c(i, 0));
		}
		a.release();
		b.release();
		c.release();

		//Forming the A matrix and B matrix
		// AA = [sum(a2.^2),  sum(a2.*ab), sum(a2.*b2), sum(a2.*ac), sum(a2.*bc)
		//     sum(a2.*ab), sum(ab. ^ 2), sum(ab.*b2), sum(ab.*ac), sum(ab.*bc)
		// 	   sum(a2.*b2), sum(ab.*b2), sum(b2. ^ 2), sum(b2.*ac), sum(b2.*bc)
		// 	   sum(a2.*ac), sum(ab.*ac), sum(b2.*ac), sum(ac. ^ 2), sum(ac.*bc)
		// 	   sum(a2.*bc), sum(ab.*bc), sum(b2.*bc), sum(ac.*bc), sum(bc. ^ 2)];
		// 
		// BB =  [sum(-(c.^ 2).*a2)
		// 		  sum(-(c.^ 2).*ab)
		// 		  sum(-(c.^ 2).*b2)
		// 		  sum(-(c.^ 2).*ac)
		// 		  sum(-(c.^ 2).*bc)];
		cv::Mat1d AA = cv::Mat::zeros(5, 5, CV_64F);
		cv::Mat1d AA_inv = cv::Mat::zeros(5, 5, CV_64F);
		cv::Mat1d BB = cv::Mat::zeros(5, 1, CV_64F);
		cv::Mat C = cv::Mat::zeros(6, 1, CV_64F);
		double BTB = 0;

		for (int i = 0; i < roi_size; ++i)
		{
			AA(0, 0) += (a2[i] * a2[i]);
			AA(0, 1) = AA(1, 0) += (a2[i] * ab[i]);
			AA(0, 2) = AA(2, 0) += (a2[i] * b2[i]);
			AA(0, 3) = AA(3, 0) += (a2[i] * ac[i]);
			AA(0, 4) = AA(4, 0) += (a2[i] * bc[i]);

			AA(1, 1) += (ab[i] * ab[i]);
			AA(1, 2) = AA(2, 1) += (ab[i] * b2[i]);
			AA(1, 3) = AA(3, 1) += (ab[i] * ac[i]);
			AA(1, 4) = AA(4, 1) += (ab[i] * bc[i]);

			AA(2, 2) += (b2[i] * b2[i]);
			AA(2, 3) = AA(3, 2) += (b2[i] * ac[i]);
			AA(2, 4) = AA(4, 2) += (b2[i] * bc[i]);

			AA(3, 3) += (ac[i] * ac[i]);
			AA(3, 4) = AA(4, 3) += (ac[i] * bc[i]);

			AA(4, 4) += (bc[i] * bc[i]);

			BB(0, 0) += (-c2[i] * a2[i]);
			BB(1, 0) += (-c2[i] * ab[i]);
			BB(2, 0) += (-c2[i] * b2[i]);
			BB(3, 0) += (-c2[i] * ac[i]);
			BB(4, 0) += (-c2[i] * bc[i]);

			BTB += (c2[i] * c2[i]);
		}

		a2.clear();
		std::vector<double>().swap(a2);
		ab.clear();
		std::vector<double>().swap(ab);
		b2.clear();
		std::vector<double>().swap(b2);
		bc.clear();
		std::vector<double>().swap(bc);
		c2.clear();
		std::vector<double>().swap(c2);
		ac.clear();
		std::vector<double>().swap(ac);

		//Solving the Least squares problem
		// X = A^-1 * B
		cv::invert(AA, AA_inv, cv::DECOMP_SVD);
		cv::Mat1d sol = AA_inv * BB;

		AA_inv.release();

		//-------------------------------------error estimation--------------------------------------//
		double stdcenter[2] = { 0 };
		cv::Mat vt, w, u;

		cv::Mat R = ((sol.t() * AA * sol) - 2 * sol.t() * BB + BTB) / (roi_size - 5);

		cv::Mat1d cvar2_constantVariance = R.at<double>(0, 0) * AA.inv();

		R.release();

		double vD = cvar2_constantVariance(3, 3);
		double vDE = cvar2_constantVariance(3, 4);
		double vE = cvar2_constantVariance(4, 4);

		cv::Mat er = cv::Mat::zeros(2, 2, CV_64F);
		er.at<double>(0, 0) = cvar2_constantVariance(3, 3); er.at<double>(0, 1) = cvar2_constantVariance(3, 4);
		er.at<double>(1, 0) = cvar2_constantVariance(4, 3); er.at<double>(1, 1) = cvar2_constantVariance(4, 4);

		cv::SVDecomp(er, w, u, vt);

		stdcenter[0] = sqrt(w.at<double>(0, 0)) / 4;
		stdcenter[1] = sqrt(w.at<double>(1, 0)) / 4;

		double angleIncertitude = atan2(vt.at<double>(1, 0), vt.at<double>(0, 0));

		if (stdcenter[0] == -1 || stdcenter[0] > 0.075) return NOT_A_PROPER_ELLIPSE;

		er.release();
		w.release();
		u.release();
		vt.release();
		cvar2_constantVariance.release();
		//----------------end of error estimation---------------------//

		AA.release();
		BB.release();

		cv::Mat1d dCnorm = cv::Mat::zeros(3, 3, CV_64F);
		dCnorm(0, 0) = sol(0, 0);
		dCnorm(0, 1) = dCnorm(1, 0) = sol(1, 0) / 2;
		dCnorm(0, 2) = dCnorm(2, 0) = sol(3, 0) / 2;
		dCnorm(1, 1) = sol(2, 0);
		dCnorm(1, 2) = dCnorm(2, 1) = sol(4, 0) / 2;
		dCnorm(2, 2) = 1;

		sol.release();

		cv::Mat1d dC;

		if (Normalization == 1)   dC = H * dCnorm * H.t();
		else dC = dCnorm;

		dCnorm.release();
		H.release();

		//The DualEllispe is found by inverting the Ellipse matrix found
		C = dC.inv();
		C /= C.at<double>(2, 2);

		dC.release();

		conic = cv::Mat::zeros(6, 1, CV_64FC1);

		conic.at<double>(0, 0) = C.at<double>(0, 0);
		conic.at<double>(1, 0) = C.at<double>(0, 1) * 2;
		conic.at<double>(2, 0) = C.at<double>(1, 1);
		conic.at<double>(3, 0) = C.at<double>(0, 2) * 2;
		conic.at<double>(4, 0) = C.at<double>(1, 2) * 2;
		conic.at<double>(5, 0) = C.at<double>(2, 2);

		C.release();

		return NO_ERROR;
	}

	// calculate gradient with Prewitt operator
	cv::Mat EllipseSolver::Prewitt(const cv::Mat& image, const string& flag)
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
					gradient.at<uchar>(row, col) = image.at<uchar>(row - 1, col - 1) + image.at<uchar>(row-1, col) + image.at<uchar>(row - 1, col + 1)
												- image.at<uchar>(row + 1, col - 1) - image.at<uchar>(row+1, col) - image.at<uchar>(row + 1, col + 1);
				}
			}
		}
		return gradient;
	}

	// calculate gradient with Sobel operator
	cv::Mat EllipseSolver::Sobel(const cv::Mat& image, const string& flag)
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

	// calculate gradient with robert operator
	cv::Mat EllipseSolver::Roberts(const cv::Mat& image, const string& flag)
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
					gradient.at<uchar>(row, col) = image.at<uchar>(row, col) - image.at<uchar>(row + 1, col + 1);
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
					gradient.at<uchar>(row, col) = image.at<uchar>(row + 1, col) - image.at<uchar>(row, col + 1);
				}
			}
		}
		return gradient;
	}

	//this Function compute the maxmum between-class variance
	int EllipseSolver::otsu(const cv::Mat& image)
	{
		int th;
		const int GrayScale = 255;					// Total Gray level of gray image
		int pixCount[GrayScale] = { 0 };			// The number of pixels occupied by each gray value
		int pixSum = image.rows * image.cols;		// The total number of pixels of the image
		float pixProportion[GrayScale] = { 0 };		// The proportion of total pixels occupied by each gray value

		float w0, w1, u0tmp, u1tmp, u0, u1, deltaTmp, deltaMax = 0;

		//ofstream out("/Users/xinxin/Desktop/pixel.txt");

		// Count the number of pixels in each gray level
		for (int i = 0; i < image.cols; i++)
		{
			for (int j = 0; j < image.rows; j++)
			{
				if (image.at<uchar>(j, i) == 0) {

					continue;
				}

				int size = image.at<uchar>(j, i);
				pixCount[image.at<uchar>(j, i)]++;

				//myout << size << " ";
			}
		}
		//myout.close();

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

	cv::Mat1d EllipseSolver::convertConicToParametric(cv::Mat& par)
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
		cv::Mat1d ell = cv::Mat::zeros(5,1,CV_64F);

		double thetarad = 0.5 * atan2(par.at<double>(1, 0), par.at<double>(0, 0) - par.at<double>(2, 0));
		double cost = cos(thetarad);
		double sint = sin(thetarad);
		double sin_squared = sint * sint;
		double cos_squared = cost * cost;
		double cos_sin = sint * cost;

		double Ao = par.at<double>(5, 0);
		double Au = par.at<double>(3, 0) * cost + par.at<double>(4, 0) * sint;
		double Av = -par.at<double>(3, 0) * sint + par.at<double>(4, 0) * cost;
		double Auu = par.at<double>(0, 0) * cos_squared + par.at<double>(2, 0) * sin_squared + par.at<double>(1, 0) * cos_sin;
		double Avv = par.at<double>(0, 0) * sin_squared + par.at<double>(2, 0) * cos_squared - par.at<double>(1, 0) * cos_sin;

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
		double angle = (thetarad/PI)*180.0;

		ell(0, 0) = centrex;
		ell(1, 0) = centrey;
		ell(2, 0) = axea;
		ell(3, 0) = axeb;
		ell(4, 0) = angle;

		/*std::cout << "centerx: " << ell.at<double>(0, 0) << endl
			<< "centery: " << ell.at<double>(1, 0) << endl
			<< "axea: " << ell.at<double>(2, 0) << endl
			<< "axeb: " << ell.at<double>(3, 0) << endl
			<< "angle: " << ell.at<double>(4, 0) << endl;*/

		return ell;
	}

	int EllipseSolver::compute(const bool& Normalization)
	{
		if (mROI.channels() == 3)
			cv::cvtColor(mROI, mGrayROI, cv::COLOR_RGB2GRAY);
		else
			mGrayROI = mROI;

		const int ImageRows = mGrayROI.rows;
		const int ImageCols = mGrayROI.cols;
		const int ImageSize = ImageRows*ImageCols;
		const float sigPixelTh = 2.0;

		cv::Mat d = (cv::Mat_<double>(1,5) << 0.2707, 0.6065, 0, -0.6065, -0.2707);
		cv::Mat g = (cv::Mat_<double>(1,5) << 0.1353, 0.6065, 1, 0.6065, 0.1353);
		
		cv::Mat opx = g.t() * d;
		cv::Mat opy = d.t() * g;

		cv::Mat GradientX, GradientY;
		cv::filter2D(mGrayROI, GradientX, CV_64F, opx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::filter2D(mGrayROI, GradientY, CV_64F, opy, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
		cv::Mat GradientROI(ImageRows, ImageCols, CV_64F,cv::Scalar(0.));

		double means = 0.0;
		for (int i = 0; i < ImageRows; i++)
		{
			for (int j = 0; j < ImageCols; j++)
			{
				double dx = GradientX.at<double>(i, j);
				double dy = GradientY.at<double>(i, j);
				GradientROI.at<double>(i, j) = sqrt(dx*dx+dy*dy);
				means += GradientROI.at<double>(i, j);
			}
		}
		if (ImageSize != 0) means /= ImageSize;

		std::vector<cv::Point_<double>> sigRoiPixel;
		for (int i = 4; i < ImageRows; i++)
		{
			for (int j = 4; j < ImageCols; j++)
			{
				if (GradientROI.at<double>(i, j) > sigPixelTh * means)
				{
					cv::Point_<float> tmp;
					tmp.y = i; tmp.x = j;
					sigRoiPixel.push_back(tmp);
				}
			}
		}

		cv::Mat conic;
		if (sigRoiPixel.size() <= 10) return NO_ELLIPSES_DETECTED;
		int err = DualConicFitting(sigRoiPixel, GradientX, GradientY, conic, Normalization);
		if (err < 0) return err;
		mPara = convertConicToParametric(conic);
		return NO_ERROR;
	}
}