#pragma once

#include <vector>
#include <iostream>
#include <list>
#include <array>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Pixel {
	unsigned int x; //图像x轴像素坐标
	unsigned int y; //图像y轴像素坐标
};
struct EdgeChains {
	std::vector<unsigned int> xCors;//边界点的x轴坐标
	std::vector<unsigned int> yCors;//边界点的y轴坐标
	std::vector<unsigned int> sId;  //边界点的像素索引
	unsigned int numOfEdges;        //边界像素数量; numOfEdges < sId.size;
};
struct LineChains {
	std::vector<unsigned int> xCors;//直线上点x坐标
	std::vector<unsigned int> yCors;//直线上点y坐标
	std::vector<unsigned int> sId;  //直线上点像素索引值
	unsigned int numOfLines;        //直线上像素点数量; numOfLines < sId.size;
};

typedef  std::list<Pixel> PixelChain;//each edge is a pixel chain


struct EDLineParam {
	int 	ksize;					//TODO高斯滤波平滑窗口
	float 	sigma;					//TODO高斯滤波平滑阈值
	float 	gradientThreshold;		//图像梯度阈值
	float 	anchorThreshold;		//提取锚点的阈值
	int 	scanIntervals;			//锚点采样分辨率
	int 	minLineLen;				//提取直线的最短像素长度
	double 	lineFitErrThreshold;	//提取直线时
	EDLineParam() {
		ksize = 5;
		sigma = 1;
	}
};

#define RELATIVE_ERROR_FACTOR   100.0
#define M_LN10   2.30258509299404568402
#define log_gamma(x)    ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*Reference:EDLines: A real-time line segment detector with a false detection control*/
/*brief: 用于从图像中提取直线信息*/
/*1、提取图像边界信息*/
/*2、提取边界信息中的直线信息*/
/*Warning:与论文中的不同之处在于像素方向发生改变时直线提取器不停止*/
class EDLineDetector
{
public:
	/*默认构造函数*/
	EDLineDetector();
	/*带配置参数的构造函数*/
	EDLineDetector(EDLineParam param);
	/*默认析构函数*/
	~EDLineDetector();
	/*提取图像中的边界*/
	/*image:	In：灰度图像;*/
	/*edges:	Out：保存边界*/
	/*smoothed: In：是否对图像进行高斯平滑*/
	/*return -1: error happen*/
	int EdgeDrawing(cv::Mat const &image, EdgeChains &edgeChains, bool smoothed = false);
	/*提取图像中的直线*/
	/*image:    In：灰度图像;*/
	/*lines:    Out：保存直线*/
	/*smoothed: In：是否对图像进行高斯平滑*/
	/*return -1: error happen*/
	int EDline(cv::Mat const &image, LineChains &lines, bool smoothed = false);
	/*提取图像中的直线*/
	/*image:    In：灰度图像;*/
	/*smoothed: In：是否对图像进行高斯平滑*/
	/*return -1: error happen*/
	int EDline(cv::Mat const &image, bool smoothed = false);

private:
	cv::Mat dxImg_;//存储x方向的梯度信息
	cv::Mat dyImg_;//存储y方向的梯度信息
	cv::Mat gImgWO_;//存储图像原始梯度信息
	LineChains lines_; //存储提取得到的直线信息
	//存储提取到的直线的直线方程[a,b,c]
	std::vector<std::array<double, 3> > lineEquations_;
	//存储提取到的直线的两个端点[x1,y1,x2,y2]
	std::vector<std::array<float, 4> > lineEndpoints_;
	//存储提取到的直线的方向
	std::vector<float>  lineDirection_;
	//存储提取到的直线是否可信的参数
	std::vector<float>  lineSalience_;
	unsigned int imageWidth;//输入图像的宽度
	unsigned int imageHeight;//输入图像的高度

private:
	/*初始化EDLine提取器*/
	void InitEDLine_();
	/*最小二乘拟合直线*/
	/*xCors:  In：像素集合的x坐标*/
	/*yCors:  In：像素集合的y坐标*/
	/*offsetS:In：像素起始索引值*/
	/*lineEquation: Out：[a,b] y=ax+b(水平方向) or x=ay+b(竖直方向)*/
	/*return: 直线拟合误差; -1:error happens*/
	double LeastSquaresLineFit_(unsigned int *xCors,unsigned int *yCors,
								unsigned int offsetS,std::array<double,2>& lineEquation);
	/*For an input pixel chain, find the best fit line. Only do the update based on new point*/
	/*For A*x=v,  Least square estimation of x = Inv(A^T * A) * (A^T * v)*/
	/*If some new observations are added, i.e, [A; A'] * x = [v; v']*/
	/*then x' = Inv(A^T * A + (A')^T * A') * (A^T * v + (A')^T * v')*/
	/*xCors:  In, pointer to the X coordinates of pixel chain*/
	/*yCors:  In, pointer to the Y coordinates of pixel chain*/
	/*offsetS:In, start index of this chain in array*/
	/*newOffsetS: In, start index of extended part*/
	/*offsetE:In, end index of this chain in array*/
	/*lineEquation: Out, [a,b] which are the coefficient of lines y=ax+b(horizontal) or x=ay+b(vertical)*/
	/*return:  line fit error; -1:error happens*/
	double LeastSquaresLineFit_(unsigned int *xCors, unsigned int *yCors,
	                            unsigned int offsetS, unsigned int newOffsetS,
	                            unsigned int offsetE,   std::array<double, 2> &lineEquation);
	/*检测提取得到的线段是否合理，基于Helmholtz principle*/
	/*xCors:	In：像素点x坐标*/
	/*yCors:	In：像素点y坐标*/
	/*offsetS:	In：起始像素点索引*/
	/*offsetE:	In：终止像素点索引*/
	/*lineEquation:	In: 直线方程*/
	/*direction:	In：直线方向*/
	/*return: true直线合格false直线不合格*/
	bool LineValidation_(unsigned int *xCors, unsigned int *yCors,
	                     unsigned int offsetS, unsigned int offsetE,
	                     std::array<double, 3> &lineEquation, float &direction);
private:
	int ksize_; //高斯滤波器卷积核大小<Default:5>
	float sigma_;//高斯滤波器卷积核方差<Default:1>
	bool bValidate_;//提取到的直线是否需要检验<Default:true>
	short gradienThreshold_;//图像梯度阈值<Default:36>
	unsigned char anchorThreshold_;//图像锚点梯度阈值<Default:8>
	unsigned int scanIntervals_;//图像锚点采样间隔<Default:2>
	int minLineLen_;//提取直线时最小直线长度阈值
	/*实例
	 *edge1 = [(7,4), (8,5), (9,6),| (10,7)|, (11, 8), (12,9)] and
	 *edge2 = [(14,9), (15,10), (16,11), (17,12),| (18, 13)|, (19,14)] ; then we store them as following:
	 *pFirstPartEdgeX_ = [10, 11, 12, 18, 19];//store the first part of each edge[from middle to end]
	 *pFirstPartEdgeY_ = [7,  8,  9,  13, 14];
	 *pFirstPartEdgeS_ = [0,3,5];// the index of start point of first part of each edge
	 *pSecondPartEdgeX_ = [10, 9, 8, 7, 18, 17, 16, 15, 14];//store the second part of each edge[from middle to front]
	 *pSecondPartEdgeY_ = [7,  6, 5, 4, 13, 12, 11, 10, 9];//anchor points(10, 7) and (18, 13) are stored again
	 *pSecondPartEdgeS_ = [0, 4, 9];// the index of start point of second part of each edge
	 *This type of storage order is because of the order of edge detection process.
	 *For each edge, start from one anchor point, first go right, then go left or first go down, then go up*/
	unsigned int *pFirstPartEdgeX_;//store the X coordinates of the first part of the pixels for chains
	unsigned int *pFirstPartEdgeY_;//store the Y coordinates of the first part of the pixels for chains
	unsigned int *pFirstPartEdgeS_;//store the start index of every edge chain in the first part arrays
	unsigned int *pSecondPartEdgeX_;//store the X coordinates of the second part of the pixels for chains
	unsigned int *pSecondPartEdgeY_;//store the Y coordinates of the second part of the pixels for chains
	unsigned int *pSecondPartEdgeS_;//store the start index of every edge chain in the second part arrays
	unsigned int *pAnchorX_;//store the X coordinates of anchors
	unsigned int *pAnchorY_;//store the Y coordinates of anchors
	cv::Mat edgeImage_;//邊緣圖像信息
	/*The threshold of line fit error;
	 *If lineFitErr is large than this threshold, then
	 *the pixel chain is not accepted as a single line segment.*/
	double lineFitErrThreshold_;

	cv::Mat gImg_;//存储图像梯度信息
	cv::Mat dirImg_;//存储图像方向信息
	double logNT_;//
	cv::Mat_<float> ATA;//the previous matrix of A^T * A;
	cv::Mat_<float> ATV;//the previous vector of A^T * V;
	cv::Mat_<float> fitMatT;//the matrix used in line fit function;
	cv::Mat_<float> fitVec;//the vector used in line fit function;
	cv::Mat_<float> tempMatLineFit;//the matrix used in line fit function;
	cv::Mat_<float> tempVecLineFit;//the vector used in line fit function;

public:
	/*判断两个double类型的数据是否相等*/
	static int double_equal(double a, double b)
	{
		double abs_diff, aa, bb, abs_max;
		if ( a == b ) return true;
		abs_diff = fabs(a - b);
		aa = fabs(a);
		bb = fabs(b);
		abs_max = aa > bb ? aa : bb;
		if ( abs_max < DBL_MIN ) abs_max = DBL_MIN;
		return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
	}
	/*使用lanczos方法计算伽马函数的自然对数值*/
	/*See:http://www.rskey.org/gamma.htm*/
	static double log_gamma_lanczos(double x)
	{
		static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
		                       8687.24529705, 1168.92649479, 83.8676043424,
		                       2.50662827511
		                     };
		double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
		double b = 0.0;
		int n;
		for (n = 0; n < 7; n++)
		{
			a -= log( x + (double) n );
			b += q[n] * pow( x, (double) n );
		}
		return a + log(b);
	}
	/*使用windschitl方法计算伽马函数的自然对数值*/
	/*See:http://www.rskey.org/gamma.htm*/
	/*Warning:x大于15时该方法效果较好*/
	static double log_gamma_windschitl(double x)
	{
		return 0.918938533204673 + (x - 0.5) * log(x) - x
		       + 0.5 * x * log( x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)) );
	}
	/** Computes -log10(NFA).
	     NFA stands for Number of False Alarms:
	     @f[
	         \mathrm{NFA} = NT \cdot B(n,k,p)
	     @f]
	     - NT       - number of tests
	     - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
	     @f[
	         B(n,k,p) = \sum_{j=k}^n
	                    \left(\begin{array}{c}n\\j\end{array}\right)
	                    p^{j} (1-p)^{n-j}
	     @f]
	     The value -log10(NFA) is equivalent but more intuitive than NFA:
	     - -1 corresponds to 10 mean false alarms
	     -  0 corresponds to 1 mean false alarm
	     -  1 corresponds to 0.1 mean false alarms
	     -  2 corresponds to 0.01 mean false alarms
	     -  ...
	     Used this way, the bigger the value, better the detection,
	     and a logarithmic scale is used.
	     @param n,k,p binomial parameters.
	     @param logNT logarithm of Number of Tests
	     The computation is based in the gamma function by the following
	     relation:
	     @f[
	         \left(\begin{array}{c}n\\k\end{array}\right)
	         = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
	     @f]
	     We use efficient algorithms to compute the logarithm of
	     the gamma function.
	     To make the computation faster, not all the sum is computed, part
	     of the terms are neglected based on a bound to the error obtained
	     (an error of 10% in the result is accepted).
	 */
	static double nfa(int n,int k,double p,double logNT)
	{
		double tolerance = 0.1;       /* an error of 10% in the result is accepted */
		double log1term, term, bin_term, mult_term, bin_tail, err, p_term;
		int i;

		//函数运行前必要的参数检查
		if ( n < 0 || k < 0 || k > n || p <= 0.0 || p >= 1.0 )
		{
			std::cout << "nfa: wrong n, k or p values." << std::endl;
			exit(0);
		}
		/* trivial cases */
		if ( n == 0 || k == 0 ) return -logNT;
		if ( n == k ) return -logNT-(double)n*log10(p);

		/* probability term */
		p_term = p / (1.0 - p);

		/* compute the first term of the series */
		/*
		  binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
		  where bincoef(n,i) are the binomial coefficients.
		  But
		    bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
		  We use this to compute the first term. Actually the log of it.
		 */
		log1term = log_gamma( (double) n + 1.0 ) - log_gamma( (double) k + 1.0 )
		           - log_gamma( (double) (n - k) + 1.0 )
		           + (double) k * log(p) + (double) (n - k) * log(1.0 - p);
		term = exp(log1term);

		/* in some cases no more computations are needed */
		if ( double_equal(term, 0.0) ) /* the first term is almost zero */
		{
			if ( (double) k > (double) n * p )    /* at begin or end of the tail?  */
				return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
			else
				return -logNT;                      /* begin: the tail is roughly 1  */
		}

		/* compute more terms if needed */
		bin_tail = term;
		for (i = k + 1; i <= n; i++)
		{
			/*    As
			    term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
			  and
			    bincoef(n,i)/bincoef(n,i-1) = n-i+1 / i,
			  then,
			    term_i / term_i-1 = (n-i+1)/i * p/(1-p)
			  and
			    term_i = term_i-1 * (n-i+1)/i * p/(1-p).
			  p/(1-p) is computed only once and stored in 'p_term'.
			 */
			bin_term = (double) (n - i + 1) / (double) i;
			mult_term = bin_term * p_term;
			term *= mult_term;
			bin_tail += term;
			if (bin_term < 1.0)
			{
				/* When bin_term<1 then mult_term_j<mult_term_i for j>i.
				  Then, the error on the binomial tail when truncated at
				  the i term can be bounded by a geometric series of form
				  term_i * sum mult_term_i^j.                            */
				err = term * ( ( 1.0 - pow( mult_term, (double) (n - i + 1) ) ) /
				               (1.0 - mult_term) - 1.0 );
				/* One wants an error at most of tolerance*final_result, or:
				  tolerance * abs(-log10(bin_tail)-logNT).
				  Now, the error that can be accepted on bin_tail is
				  given by tolerance*final_result divided by the derivative
				  of -log10(x) when x=bin_tail. that is:
				  tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
				  Finally, we truncate the tail if the error is less than:
				  tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
				if ( err < tolerance * fabs(-log10(bin_tail) - logNT) * bin_tail ) break;
			}
		}
		return -log10(bin_tail) - logNT;
	}
};
