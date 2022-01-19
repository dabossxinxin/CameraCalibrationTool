#include <fstream>
#include <iostream>
#include "EDLineDetector.h"

//#define DEBUG_EDLINE

#ifdef DEBUG_EDLINE
	#include <sys/stat.h>
	#include <sys/types.h>
	#include <ctime>
	#include <sys/errno.h>
	static std::string DEBUG_FOLDER;
#endif

#define Horizontal			255//if |dx|<|dy|;
#define Vertical			0//if |dy|<=|dx|;
#define UpDir				1
#define RightDir			2
#define DownDir				3
#define LeftDir				4
#define TryTime				6
#define SkipEdgePoint		2
#define M_PI				3.1415926535898

void writeMat(cv::Mat m, std::string name, int n)
{
	std::stringstream ss;
	std::string s;
	ss << n;
	ss >> s;
	std::string fileNameConf = name + s;
	cv::FileStorage fsConf(fileNameConf, cv::FileStorage::WRITE);
	fsConf << "m" << m;
	fsConf.release();
}


/**
 * @brief writeImage save cv::Mat to image(special for 32CV_32F data type)
 *
 * @param img  image to save
 * @param name image name
 */
void writeImage(cv::Mat img, std::string name)
{
	if(img.depth() == CV_32F)
	{
		double minVal, maxVal, alpha, beta;
		cv::minMaxLoc(img, &minVal, &maxVal);
		if(minVal >= maxVal)
		{
			return;
		}
		alpha = UINT16_MAX/(maxVal-minVal);
		beta = -minVal*alpha;
		cv::Mat img8;
		img.convertTo(img8, CV_16UC1, alpha, beta);
		cv::imwrite(name, img8);
	}
	else
	{
		cv::imwrite(name, img);
	}
}


EDLineDetector::EDLineDetector()
{
	//  cout<<"Call EDLineDetector constructor function"<<endl;
	//set parameters for line segment detection
	ksize_ = 5;
	sigma_ = 1.0;
	gradienThreshold_ = 80; // ***** ORIGINAL WAS 25
	anchorThreshold_  = 2;//8
	scanIntervals_    = 2;//2
	minLineLen_       = 15;
	lineFitErrThreshold_ = 1.4;
	InitEDLine_();
}

EDLineDetector::EDLineDetector(EDLineParam param)
{
	//set parameters for line segment detection
	ksize_ = param.ksize;
	sigma_ = param.sigma;
	gradienThreshold_ = param.gradientThreshold;
	anchorThreshold_  = param.anchorThreshold;
	scanIntervals_    = param.scanIntervals;
	minLineLen_       = param.minLineLen;
	lineFitErrThreshold_ = param.lineFitErrThreshold;

	InitEDLine_();
}

void EDLineDetector::InitEDLine_()
{

#ifdef DEBUG_EDLINE_LOG
	fs.open("edline_log.yml", cv::FileStorage::WRITE);
#endif

	bValidate_           = true;
	ATA = cv::Mat_<int>(2, 2);
	ATV = cv::Mat_<int>(1, 2);
	tempMatLineFit = cv::Mat_<int>(2, 2);
	tempVecLineFit = cv::Mat_<int>(1, 2);
	fitMatT = cv::Mat_<int>(2, minLineLen_);
	fitVec = cv::Mat_<int>(1, minLineLen_);

	for (int i = 0; i < minLineLen_; i++)
	{
		fitMatT[1][i] = 1;
	}

	dxImg_.create( 1, 1, CV_32FC1 );
	dyImg_.create( 1, 1, CV_32FC1 );
	gImgWO_.create( 1, 1, CV_32FC1 );
	pFirstPartEdgeX_ = NULL;
	pFirstPartEdgeY_ = NULL;
	pFirstPartEdgeS_ = NULL;
	pSecondPartEdgeX_ = NULL;
	pSecondPartEdgeY_ = NULL;
	pSecondPartEdgeS_ = NULL;
	pAnchorX_ = NULL;
	pAnchorY_ = NULL;
}

EDLineDetector::~EDLineDetector()
{
	if (pFirstPartEdgeX_ != NULL)
	{
		delete [] pFirstPartEdgeX_;
		delete [] pFirstPartEdgeY_;
		delete [] pSecondPartEdgeX_;
		delete [] pSecondPartEdgeY_;
		delete [] pAnchorX_;
		delete [] pAnchorY_;
	}

	if (pFirstPartEdgeS_ != NULL)
	{
		delete [] pFirstPartEdgeS_;
		delete [] pSecondPartEdgeS_;
	}


}

int EDLineDetector::EdgeDrawing(cv::Mat const &image, EdgeChains &edgeChains, bool smoothed )
{
	imageWidth = image.cols;
	imageHeight = image.rows;
	unsigned int pixelNum = imageWidth * imageHeight;

	cv::Mat cImage = image.clone();

	if (!smoothed) //input image hasn't been smoothed.
	{
		cv::GaussianBlur(image, cImage, cv::Size(ksize_, ksize_), sigma_);
	}

	//Is this useful? Doesn't seem to have references
	//else{
//		unsigned char *pImage = image.GetImageData();
//		memcpy(cvImage->data.ptr, pImage, pixelNum*sizeof(unsigned char));
//	}

	unsigned int edgePixelArraySize = pixelNum / 5;
	unsigned int maxNumOfEdge = edgePixelArraySize / 20;
	//compute dx, dy images
	//
	if (gImg_.cols != imageWidth || gImg_.rows != imageHeight)
	{
		if (pFirstPartEdgeX_ != NULL)
		{
			delete [] pFirstPartEdgeX_;
			delete [] pFirstPartEdgeY_;
			delete [] pSecondPartEdgeX_;
			delete [] pSecondPartEdgeY_;
			delete [] pFirstPartEdgeS_;
			delete [] pSecondPartEdgeS_;
			delete [] pAnchorX_;
			delete [] pAnchorY_;
		}

		dxImg_.create(imageHeight, imageWidth, CV_32FC1);
		dyImg_.create(imageHeight, imageWidth, CV_32FC1 );
		gImgWO_.create(imageHeight, imageWidth, CV_32FC1 );
		gImg_.create(imageHeight, imageWidth, CV_32FC1 );
		dirImg_.create(imageHeight, imageWidth, CV_8UC1 );
		edgeImage_.create(imageHeight, imageWidth, CV_8UC1 );

		pFirstPartEdgeX_ = new unsigned int[edgePixelArraySize];
		pFirstPartEdgeY_ = new unsigned int[edgePixelArraySize];
		pSecondPartEdgeX_ = new unsigned int[edgePixelArraySize];
		pSecondPartEdgeY_ = new unsigned int[edgePixelArraySize];
		pFirstPartEdgeS_ = new unsigned int[maxNumOfEdge];
		pSecondPartEdgeS_ = new unsigned int[maxNumOfEdge];
		pAnchorX_ = new unsigned int[edgePixelArraySize];
		pAnchorY_ = new unsigned int[edgePixelArraySize];
	}

	cv::Sobel( cImage, dxImg_, CV_32FC1, 1, 0, 3);
	cv::Sobel( cImage, dyImg_, CV_32FC1, 0, 1, 3);

	//compute gradient and direction images

	cv::Mat sumDxDy;
	double t = (double)cv::getTickCount();

	cv::Mat dxABS_m = cv::abs(dxImg_);
	cv::Mat dyABS_m = cv::abs(dyImg_);
	cv::add(dyABS_m, dxABS_m, sumDxDy);
	cv::threshold(sumDxDy, gImg_, gradienThreshold_ + 1, 0, cv::THRESH_TOZERO);
	gImg_ = gImg_ / 4;
	gImgWO_ = sumDxDy / 4;
	cv::compare(dxABS_m, dyABS_m, dirImg_, cv::CMP_LT);


//	cv::imwrite("sumDxDy.png", sumDxDy);

#ifdef DEBUG_EDLINE
	DEBUG_FOLDER = std::string("./") + std::to_string(std::time(0));
	int status = mkdir(DEBUG_FOLDER.c_str(), 0777);
	if(!status)
	{
//        cv::imshow("dxImg_", dxImg_);
//        cv::imshow("dyImg_", dyImg_);
//        cv::waitKey();
		writeImage(dxImg_, DEBUG_FOLDER+"/dxImg.png");
		writeImage(dyImg_, DEBUG_FOLDER+"/dyImg.png");
		writeImage(gImg_, DEBUG_FOLDER+"/gImg.png");
		writeImage(dirImg_, DEBUG_FOLDER+"/dirImg.png");
	}
	else
	{
		DEBUG_FOLDER.clear();
		std::cerr << "Fail to create folder: " << DEBUG_FOLDER << std::endl
				  << "Error No: " << errno << std::endl
				  << strerror(errno) << std::endl;
	}
#endif

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	//std::cout << "FOR ABS: " << t << "s" << std::endl;

	//short *pdxImg = dxImg_.ptr<short>();
	//short *pdyImg = dyImg_.ptr<short>();

	float *pgImg  = gImg_.ptr<float>();
	float gValue1, gValue2, gValue3;

	unsigned char *pdirImg = dirImg_.ptr();

	//extract the anchors in the gradient image, store into a vector
	memset(pAnchorX_,  0, edgePixelArraySize * sizeof(unsigned int)); //initialization
	memset(pAnchorY_,  0, edgePixelArraySize * sizeof(unsigned int));
	unsigned int anchorsSize = 0;
	unsigned int indexInArray;

	for (unsigned int w = 1; w < imageWidth - 1; w = w + scanIntervals_)
	{
		for (unsigned int h = 1; h < imageHeight - 1; h = h + scanIntervals_)
		{
			indexInArray = h * imageWidth + w;
			//gValue1 = pdirImg[indexInArray];
			if (pdirImg[indexInArray] == Horizontal) //if the direction of pixel is horizontal, then compare with up and down
			{
				//gValue2 = pgImg[indexInArray];
				if (pgImg[indexInArray] >= pgImg[indexInArray - imageWidth] + anchorThreshold_
						&& pgImg[indexInArray] >= pgImg[indexInArray + imageWidth] + anchorThreshold_) // (w,h) is accepted as an anchor
				{
					pAnchorX_[anchorsSize]   = w;
					pAnchorY_[anchorsSize++] = h;
				}
			}
			else  // if(pdirImg[indexInArray]==Vertical){//it is vertical edge, should be compared with left and right
			{
				//gValue2 = pgImg[indexInArray];
				if (pgImg[indexInArray] >= pgImg[indexInArray - 1] + anchorThreshold_
						&& pgImg[indexInArray] >= pgImg[indexInArray + 1] + anchorThreshold_) // (w,h) is accepted as an anchor
				{
					pAnchorX_[anchorsSize]   = w;
					pAnchorY_[anchorsSize++] = h;
				}
			}
		}
	}

	if (anchorsSize > edgePixelArraySize)
	{
		std::cout << "anchor size is larger than its maximal size. anchorsSize=" << anchorsSize
			 << ", maximal size = " << edgePixelArraySize << std::endl;
		return -1;
	}

#ifdef DEBUG_EDLINE
	cout << "Anchor point detection, anchors.size=" << anchorsSize << endl;
#endif
	//link the anchors by smart routing
	edgeImage_.setTo(0);
	unsigned char *pEdgeImg = edgeImage_.data;
	memset(pFirstPartEdgeX_,  0, edgePixelArraySize * sizeof(unsigned int)); //initialization
	memset(pFirstPartEdgeY_,  0, edgePixelArraySize * sizeof(unsigned int));
	memset(pSecondPartEdgeX_, 0, edgePixelArraySize * sizeof(unsigned int));
	memset(pSecondPartEdgeY_, 0, edgePixelArraySize * sizeof(unsigned int));
	memset(pFirstPartEdgeS_,  0, maxNumOfEdge * sizeof(unsigned int));
	memset(pSecondPartEdgeS_, 0, maxNumOfEdge * sizeof(unsigned int));
	unsigned int offsetPFirst = 0, offsetPSecond = 0;
	unsigned int offsetPS = 0;

	unsigned int x, y;
	unsigned int lastX, lastY;
	unsigned char lastDirection;//up = 1, right = 2, down = 3, left = 4;
	unsigned char shouldGoDirection;//up = 1, right = 2, down = 3, left = 4;
	int edgeLenFirst, edgeLenSecond;
	for (unsigned int i = 0; i < anchorsSize; i++)
	{
		x = pAnchorX_[i];
		y = pAnchorY_[i];
		indexInArray = y * imageWidth + x;
		if (pEdgeImg[indexInArray]) //if anchor i is already been an edge pixel.
		{
			continue;
		}
		/*The walk stops under 3 conditions:
		 * 1. We move out of the edge areas, i.e., the thresholded gradient value
		 *    of the current pixel is 0.
		 * 2. The current direction of the edge changes, i.e., from horizontal
		 *    to vertical or vice versa.?? (This is turned out not correct. From the online edge draw demo
		 *    we can figure out that authors don't implement this rule either because their extracted edge
		 *    chain could be a circle which means pixel directions would definitely be different
		 *    in somewhere on the chain.)
		 * 3. We encounter a previously detected edge pixel. */
		pFirstPartEdgeS_[offsetPS] = offsetPFirst;
		if (pdirImg[indexInArray] == Horizontal) //if the direction of this pixel is horizontal, then go left and right.
		{
			//fist go right, pixel direction may be different during linking.
			lastDirection = RightDir;
			while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray])
			{
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
				pFirstPartEdgeX_[offsetPFirst] = x;
				pFirstPartEdgeY_[offsetPFirst++] = y;
				shouldGoDirection = 0;//unknown
				if (pdirImg[indexInArray] == Horizontal) //should go left or right
				{
					if (lastDirection == UpDir || lastDirection == DownDir) //change the pixel direction now
					{
						if (x > lastX) //should go right
						{
							shouldGoDirection = RightDir;
						}
						else  //should go left
						{
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == RightDir || shouldGoDirection == RightDir) //go right
					{
						if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray + 1];
						gValue3 = pgImg[indexInArray + imageWidth + 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else  //straight-right
						{
							x = x + 1;
						}
						lastDirection = RightDir;
					}
					else if (lastDirection == LeftDir || shouldGoDirection == LeftDir) //go left
					{
						if (x == 0 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth - 1];
						gValue2 = pgImg[indexInArray - 1];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-left
						{
							x = x - 1;
						}
						lastDirection = LeftDir;
					}
				}
				else  //should go up or down.
				{
					if (lastDirection == RightDir || lastDirection == LeftDir) //change the pixel direction now
					{
						if (y > lastY) //should go down
						{
							shouldGoDirection = DownDir;
						}
						else  //should go up
						{
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == DownDir || shouldGoDirection == DownDir) //go down
					{
						if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray + imageWidth + 1];
						gValue2 = pgImg[indexInArray + imageWidth];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-down
						{
							y = y + 1;
						}
						lastDirection = DownDir;
					}
					else if (lastDirection == UpDir || shouldGoDirection == UpDir) //go up
					{
						if (x == 0 || x == imageWidth - 1 || y == 0) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray - imageWidth];
						gValue3 = pgImg[indexInArray - imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else  //straight-up
						{
							y = y - 1;
						}
						lastDirection = UpDir;
					}
				}
				indexInArray = y * imageWidth + x;
			}//end while go right
			//then go left, pixel direction may be different during linking.
			x = pAnchorX_[i];
			y = pAnchorY_[i];
			indexInArray = y * imageWidth + x;
			pEdgeImg[indexInArray] = 0;//mark the anchor point be a non-edge pixel and
			lastDirection = LeftDir;
			pSecondPartEdgeS_[offsetPS] = offsetPSecond;
			while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray])
			{
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
				pSecondPartEdgeX_[offsetPSecond] = x;
				pSecondPartEdgeY_[offsetPSecond++] = y;
				shouldGoDirection = 0;//unknown
				if (pdirImg[indexInArray] == Horizontal) //should go left or right
				{
					if (lastDirection == UpDir || lastDirection == DownDir) //change the pixel direction now
					{
						if (x > lastX) //should go right
						{
							shouldGoDirection = RightDir;
						}
						else  //should go left
						{
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == RightDir || shouldGoDirection == RightDir) //go right
					{
						if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray + 1];
						gValue3 = pgImg[indexInArray + imageWidth + 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else  //straight-right
						{
							x = x + 1;
						}
						lastDirection = RightDir;
					}
					else   if (lastDirection == LeftDir || shouldGoDirection == LeftDir) //go left
					{
						if (x == 0 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth - 1];
						gValue2 = pgImg[indexInArray - 1];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-left
						{
							x = x - 1;
						}
						lastDirection = LeftDir;
					}
				}
				else  //should go up or down.
				{
					if (lastDirection == RightDir || lastDirection == LeftDir) //change the pixel direction now
					{
						if (y > lastY) //should go down
						{
							shouldGoDirection = DownDir;
						}
						else  //should go up
						{
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == DownDir || shouldGoDirection == DownDir) //go down
					{
						if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray + imageWidth + 1];
						gValue2 = pgImg[indexInArray + imageWidth];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-down
						{
							y = y + 1;
						}
						lastDirection = DownDir;
					}
					else   if (lastDirection == UpDir || shouldGoDirection == UpDir) //go up
					{
						if (x == 0 || x == imageWidth - 1 || y == 0) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray - imageWidth];
						gValue3 = pgImg[indexInArray - imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else  //straight-up
						{
							y = y - 1;
						}
						lastDirection = UpDir;
					}
				}
				indexInArray = y * imageWidth + x;
			}//end while go left
			//end anchor is Horizontal
		}
		else  //the direction of this pixel is vertical, go up and down
		{
			//fist go down, pixel direction may be different during linking.
			lastDirection = DownDir;
			while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray])
			{
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
				pFirstPartEdgeX_[offsetPFirst] = x;
				pFirstPartEdgeY_[offsetPFirst++] = y;
				shouldGoDirection = 0;//unknown
				if (pdirImg[indexInArray] == Horizontal) //should go left or right
				{
					if (lastDirection == UpDir || lastDirection == DownDir) //change the pixel direction now
					{
						if (x > lastX) //should go right
						{
							shouldGoDirection = RightDir;
						}
						else  //should go left
						{
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == RightDir || shouldGoDirection == RightDir) //go right
					{
						if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray + 1];
						gValue3 = pgImg[indexInArray + imageWidth + 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else  //straight-right
						{
							x = x + 1;
						}
						lastDirection = RightDir;
					}
					else   if (lastDirection == LeftDir || shouldGoDirection == LeftDir) //go left
					{
						if (x == 0 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth - 1];
						gValue2 = pgImg[indexInArray - 1];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-left
						{
							x = x - 1;
						}
						lastDirection = LeftDir;
					}
				}
				else  //should go up or down.
				{
					if (lastDirection == RightDir || lastDirection == LeftDir) //change the pixel direction now
					{
						if (y > lastY) //should go down
						{
							shouldGoDirection = DownDir;
						}
						else  //should go up
						{
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == DownDir || shouldGoDirection == DownDir) //go down
					{
						if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray + imageWidth + 1];
						gValue2 = pgImg[indexInArray + imageWidth];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-down
						{
							y = y + 1;
						}
						lastDirection = DownDir;
					}
					else   if (lastDirection == UpDir || shouldGoDirection == UpDir) //go up
					{
						if (x == 0 || x == imageWidth - 1 || y == 0) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray - imageWidth];
						gValue3 = pgImg[indexInArray - imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else  //straight-up
						{
							y = y - 1;
						}
						lastDirection = UpDir;
					}
				}
				indexInArray = y * imageWidth + x;
			}//end while go down
			//then go up, pixel direction may be different during linking.
			lastDirection = UpDir;
			x = pAnchorX_[i];
			y = pAnchorY_[i];
			indexInArray = y * imageWidth + x;
			pEdgeImg[indexInArray] = 0;//mark the anchor point be a non-edge pixel and
			pSecondPartEdgeS_[offsetPS] = offsetPSecond;
			while (pgImg[indexInArray] > 0 && !pEdgeImg[indexInArray])
			{
				pEdgeImg[indexInArray] = 1;        // Mark this pixel as an edge pixel
				pSecondPartEdgeX_[offsetPSecond] = x;
				pSecondPartEdgeY_[offsetPSecond++] = y;
				shouldGoDirection = 0;//unknown
				if (pdirImg[indexInArray] == Horizontal) //should go left or right
				{
					if (lastDirection == UpDir || lastDirection == DownDir) //change the pixel direction now
					{
						if (x > lastX) //should go right
						{
							shouldGoDirection = RightDir;
						}
						else  //should go left
						{
							shouldGoDirection = LeftDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == RightDir || shouldGoDirection == RightDir) //go right
					{
						if (x == imageWidth - 1 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the right and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray + 1];
						gValue3 = pgImg[indexInArray + imageWidth + 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else  //straight-right
						{
							x = x + 1;
						}
						lastDirection = RightDir;
					}
					else   if (lastDirection == LeftDir || shouldGoDirection == LeftDir) //go left
					{
						if (x == 0 || y == 0 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the left and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth - 1];
						gValue2 = pgImg[indexInArray - 1];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-left
						{
							x = x - 1;
						}
						lastDirection = LeftDir;
					}
				}
				else  //should go up or down.
				{
					if (lastDirection == RightDir || lastDirection == LeftDir) //change the pixel direction now
					{
						if (y > lastY) //should go down
						{
							shouldGoDirection = DownDir;
						}
						else  //should go up
						{
							shouldGoDirection = UpDir;
						}
					}
					lastX = x;
					lastY = y;
					if (lastDirection == DownDir || shouldGoDirection == DownDir) //go down
					{
						if (x == 0 || x == imageWidth - 1 || y == imageHeight - 1) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the down and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray + imageWidth + 1];
						gValue2 = pgImg[indexInArray + imageWidth];
						gValue3 = pgImg[indexInArray + imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //down-right
						{
							x = x + 1;
							y = y + 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //down-left
						{
							x = x - 1;
							y = y + 1;
						}
						else  //straight-down
						{
							y = y + 1;
						}
						lastDirection = DownDir;
					}
					else   if (lastDirection == UpDir || shouldGoDirection == UpDir) //go up
					{
						if (x == 0 || x == imageWidth - 1 || y == 0) //reach the image border
						{
							break;
						}
						// Look at 3 neighbors to the up and pick the one with the max. gradient value
						gValue1 = pgImg[indexInArray - imageWidth + 1];
						gValue2 = pgImg[indexInArray - imageWidth];
						gValue3 = pgImg[indexInArray - imageWidth - 1];
						if (gValue1 >= gValue2 && gValue1 >= gValue3) //up-right
						{
							x = x + 1;
							y = y - 1;
						}
						else if (gValue3 >= gValue2 && gValue3 >= gValue1) //up-left
						{
							x = x - 1;
							y = y - 1;
						}
						else  //straight-up
						{
							y = y - 1;
						}
						lastDirection = UpDir;
					}
				}
				indexInArray = y * imageWidth + x;
			}//end while go up
		}//end anchor is Vertical
		//only keep the edge chains whose length is larger than the minLineLen_;
		edgeLenFirst = offsetPFirst - pFirstPartEdgeS_[offsetPS];
		edgeLenSecond = offsetPSecond - pSecondPartEdgeS_[offsetPS];
		if (edgeLenFirst + edgeLenSecond < minLineLen_ + 1) //short edge, drop it
		{
			offsetPFirst = pFirstPartEdgeS_[offsetPS];
			offsetPSecond = pSecondPartEdgeS_[offsetPS];
		}
		else
		{
			offsetPS++;
		}
	}
	//store the last index
	pFirstPartEdgeS_[offsetPS]  = offsetPFirst;
	pSecondPartEdgeS_[offsetPS] = offsetPSecond;
	if (offsetPS > maxNumOfEdge)
	{
		std::cout << "Edge drawing Error: The total number of edges is larger than MaxNumOfEdge, "
			 "numofedge = " << offsetPS << ", MaxNumOfEdge=" << maxNumOfEdge << std::endl;
		return -1;
	}
	if (offsetPFirst > edgePixelArraySize || offsetPSecond > edgePixelArraySize)
	{
		std::cout << "Edge drawing Error: The total number of edge pixels is larger than MaxNumOfEdgePixels, "
			 "numofedgePixel1 = " << offsetPFirst << ",  numofedgePixel2 = " << offsetPSecond <<
			 ", MaxNumOfEdgePixel=" << edgePixelArraySize << std::endl;
		return -1;
	}

	/*now all the edge information are stored in pFirstPartEdgeX_, pFirstPartEdgeY_,
	 *pFirstPartEdgeS_,  pSecondPartEdgeX_, pSecondPartEdgeY_, pSecondPartEdgeS_;
	 *we should reorganize them into edgeChains for easily using.   */
	//unsigned int tempID;
	long  int tempID;
	edgeChains.xCors.resize(offsetPFirst + offsetPSecond);
	edgeChains.yCors.resize(offsetPFirst + offsetPSecond);
	edgeChains.sId.resize(offsetPS + 1);
	unsigned int *pxCors = edgeChains.xCors.data();
	unsigned int *pyCors = edgeChains.yCors.data();
	unsigned int *psId   = edgeChains.sId.data();
	offsetPFirst = 0;
	offsetPSecond = 0;
	unsigned int indexInCors = 0;
	unsigned int numOfEdges = 0;
	for (unsigned int edgeId = 0; edgeId < offsetPS; edgeId++)
	{
		//step1, put the first and second parts edge coordinates together from edge start to edge end
		psId[numOfEdges++] = indexInCors;
		indexInArray = pFirstPartEdgeS_[edgeId];
		offsetPFirst = pFirstPartEdgeS_[edgeId + 1];
		for (tempID = offsetPFirst - 1; tempID >= indexInArray; tempID--) //add first part edge
		{
			if (tempID < 0) break;
			pxCors[indexInCors]   = pFirstPartEdgeX_[tempID];
			pyCors[indexInCors++] = pFirstPartEdgeY_[tempID];
		}
		indexInArray = pSecondPartEdgeS_[edgeId];
		offsetPSecond = pSecondPartEdgeS_[edgeId + 1];
		for (tempID = indexInArray + 1; tempID < offsetPSecond; tempID++) //add second part edge
		{
			if (tempID >= offsetPSecond) break;
			pxCors[indexInCors]  = pSecondPartEdgeX_[tempID];
			pyCors[indexInCors++] = pSecondPartEdgeY_[tempID];
		}
	}
	psId[numOfEdges] = indexInCors;//the end index of the last edge
	edgeChains.numOfEdges = numOfEdges;

#ifdef DEBUG_EDLINE
	if(!DEBUG_FOLDER.empty())
	{
		std::string edge_folder = DEBUG_FOLDER+"/edge/";
		int status = mkdir(edge_folder.c_str(), 0777);

		if(!status)
		{
			/*Show the extracted edge cvImage in color. Each chain is in different color.*/
			cout << "Edge Drawing, numofedge = " << numOfEdges << endl;
			cv::Mat tmp, singleImg;
			cv::cvtColor(dirImg_, tmp, CV_GRAY2BGR);
			cv::cvtColor(dirImg_, singleImg, CV_GRAY2BGR);
			cv::Vec3b rColor;
			int lowest = 100, highest = 255;
			int range = (highest - lowest) + 1;

			srand((unsigned)time(0));

			for (unsigned int i = 0; i < edgeChains.numOfEdges; i++)
			{
				cv::Mat foo = singleImg.clone();
				std::ofstream f(edge_folder+std::to_string(i)+".txt");

				rColor.val[0] = lowest + int(rand() % range);
				rColor.val[1] = lowest + int(rand() % range);
				rColor.val[2] = lowest + int(rand() % range);

				for (indexInCors = psId[i]; indexInCors < psId[i + 1]; indexInCors++)
				{
					tmp.at<cv::Vec3b>(pyCors[indexInCors], pxCors[indexInCors])	= rColor;
					foo.at<cv::Vec3b>(pyCors[indexInCors], pxCors[indexInCors])	= cv::Vec3b{0x00,0x00,0xff};
					f << pyCors[indexInCors] << " , " << pxCors[indexInCors] << std::endl;
				}

				cv::imwrite(edge_folder+std::to_string(i)+".png",foo);
				f.close();
			}
			cv::imwrite(DEBUG_FOLDER+"/edges.png", tmp);

		}
		else
		{
			std::cerr << "Fail to create edge folder: " << edge_folder << std::endl
					  << "Error No: " << errno << std::endl
					  << strerror(errno) << std::endl;
		}

	}
#endif

	return 1;
}


int EDLineDetector::EDline(cv::Mat const &image, LineChains &lines, bool smoothed)
{

	//first, call EdgeDrawing function to extract edges
	EdgeChains edges;
	if ( EdgeDrawing(image, edges, smoothed) < 0 )
	{
		std::cout << "Line Detection not finished" << std::endl;
		return -1;
	}

	//  bValidate_ =false;

	//detect lines
	unsigned int linePixelID = edges.sId[edges.numOfEdges];
	lines.xCors.resize(linePixelID);
	lines.yCors.resize(linePixelID);
	lines.sId.resize(5 * edges.numOfEdges);

	unsigned int *pEdgeXCors = edges.xCors.data();
	unsigned int *pEdgeYCors = edges.yCors.data();
	unsigned int *pEdgeSID   = edges.sId.data();
	unsigned int *pLineXCors = lines.xCors.data();
	unsigned int *pLineYCors = lines.yCors.data();
	unsigned int *pLineSID   = lines.sId.data();
	unsigned char *pdirImg = dirImg_.data;
	unsigned int numOfLines = 0;
	unsigned int offsetInEdgeArrayS, offsetInEdgeArrayE, newOffsetS;//start index and end index
	unsigned int offsetInLineArray = 0;
	float direction;//line direction
	double lineFitErr;//the line fit error;
	std::array<double, 2> lineEquation;

	logNT_ = 2.0 * ( log10( (double) imageWidth ) + log10( (double) imageHeight ) );
	lineEquations_.clear();
	lineEndpoints_.clear();
	lineDirection_.clear();

	for (unsigned int edgeID = 0; edgeID < edges.numOfEdges; edgeID++)
	{
		offsetInEdgeArrayS = pEdgeSID[edgeID];
		offsetInEdgeArrayE = pEdgeSID[edgeID + 1];

		//extract line segments from an edge, may find more than one segments
		while (offsetInEdgeArrayE > offsetInEdgeArrayS + minLineLen_)
		{
			//find an initial line segment
			while (offsetInEdgeArrayE > offsetInEdgeArrayS + minLineLen_)
			{
				lineFitErr = LeastSquaresLineFit_(pEdgeXCors, pEdgeYCors,
												  offsetInEdgeArrayS, lineEquation);
				if ( lineFitErr <= lineFitErrThreshold_ )
				{
					break; //ok, an initial line segment detected
				}
				//skip the first two pixel in the chain and try with the remaining pixels
				offsetInEdgeArrayS += SkipEdgePoint;
			}
			// exit if no initial line is detected
			if (lineFitErr > lineFitErrThreshold_)
			{
				break;
			}

			// An initial line segment is detected. Try to extend this line segment
			pLineSID[numOfLines] = offsetInLineArray;
			// for a line ax+by+c=0, coef1 = 1/sqrt(a^2+b^2);
			double coef1;
			// for a line ax+by+c=0 and a point(xi, yi),
			// pointToLineDis = coef1*|a*xi+b*yi+c|
			double pointToLineDis;
			bool bExtended = true;
			bool bFirstTry = true;
			//to against noise, we accept a few outlier of a line.
			int numOfOutlier;
			int tryTimes = 0;

			//y=ax+b, i.e. ax-y+b=0
			if ( pdirImg[pEdgeYCors[offsetInEdgeArrayS]*imageWidth + pEdgeXCors[offsetInEdgeArrayS]]
					== Horizontal )
			{
				while (bExtended)
				{
					tryTimes++;
					if (bFirstTry)
					{
						bFirstTry = false;
						//First add the initial line segment to the line array
						for (int i = 0; i < minLineLen_; i++)
						{
							pLineXCors[offsetInLineArray]   = pEdgeXCors[offsetInEdgeArrayS];
							pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
						}
					}
					else  //after each try, line is extended, line equation should be re-estimated
					{
						//adjust the line equation
						lineFitErr = LeastSquaresLineFit_(pLineXCors, pLineYCors, pLineSID[numOfLines],
														  newOffsetS, offsetInLineArray, lineEquation);
					}

					coef1 = 1 / sqrt( lineEquation[0]*lineEquation[0] + 1);
					numOfOutlier = 0;
					newOffsetS = offsetInLineArray;

					while (offsetInEdgeArrayE > offsetInEdgeArrayS)
					{
						pointToLineDis = fabs(lineEquation[0] * pEdgeXCors[offsetInEdgeArrayS] -
											  pEdgeYCors[offsetInEdgeArrayS] + lineEquation[1]) * coef1;
						pLineXCors[offsetInLineArray] = pEdgeXCors[offsetInEdgeArrayS];
						pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
						if (pointToLineDis > lineFitErrThreshold_)
						{
							numOfOutlier++;
							if (numOfOutlier > 3) break;
						}
						else  //we count number of connective outliers.
						{
							numOfOutlier = 0;
						}
					}
					//pop back the last few outliers from lines and return them to edge chain
					offsetInLineArray  -= numOfOutlier;
					offsetInEdgeArrayS -= numOfOutlier;

					//some new pixels are added to the line
//					if (offsetInLineArray - newOffsetS > 0 && tryTimes < TryTime)
//					{
//					}
//					else
					if (offsetInLineArray <= newOffsetS || tryTimes >= TryTime)
					{
						bExtended = false;//no new pixels are added.
					}
				}
				//the line equation coefficients,for line w1x+w2y+w3 =0, we normalize it to make w1^2+w2^2 = 1.
				std::array<double, 3> lineEqu {{lineEquation[0]*coef1, -1 * coef1, lineEquation[1]*coef1}};
				//check the line
				if ( LineValidation_(pLineXCors, pLineYCors, pLineSID[numOfLines],
									 offsetInLineArray, lineEqu, direction) )
				{
					//store the line equation coefficients
					lineEquations_.push_back(lineEqu);
					/*At last, compute the line endpoints and store them.
					 *we project the first and last pixels in the pixelChain onto the best fit line
					 *to get the line endpoints.
					 *xp= (w2^2*x0-w1*w2*y0-w3*w1)/(w1^2+w2^2)
					 *yp= (w1^2*y0-w1*w2*x0-w3*w2)/(w1^2+w2^2)  */
					std::array<float, 4> lineEndP;//line endpoints
					double a1 = lineEqu[1] * lineEqu[1];
					double a2 = lineEqu[0] * lineEqu[0];
					double a3 = lineEqu[0] * lineEqu[1];
					double a4 = lineEqu[2] * lineEqu[0];
					double a5 = lineEqu[2] * lineEqu[1];
					unsigned int Px = pLineXCors[pLineSID[numOfLines] ];//first pixel
					unsigned int Py = pLineYCors[pLineSID[numOfLines] ];
					lineEndP[0] = a1 * Px - a3 * Py - a4; //x
					lineEndP[1] = a2 * Py - a3 * Px - a5; //y
					Px = pLineXCors[offsetInLineArray - 1 ]; //last pixel
					Py = pLineYCors[offsetInLineArray - 1 ];
					lineEndP[2] = a1 * Px - a3 * Py - a4; //x
					lineEndP[3] = a2 * Py - a3 * Px - a5; //y
					lineEndpoints_.push_back(lineEndP);
					lineDirection_.push_back(direction);
					numOfLines++;
				}
				else
				{
					offsetInLineArray = pLineSID[numOfLines];// line was not accepted, the offset is set back
				}
			}
			else  //x=ay+b, i.e. x-ay-b=0
			{
				while (bExtended)
				{
					tryTimes++;
					if (bFirstTry)
					{
						bFirstTry = false;
						for (int i = 0; i < minLineLen_; i++) //First add the initial line segment to the line array
						{
							pLineXCors[offsetInLineArray]   = pEdgeXCors[offsetInEdgeArrayS];
							pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
						}
					}
					else  //after each try, line is extended, line equation should be re-estimated
					{
						//adjust the line equation
						lineFitErr = LeastSquaresLineFit_(pLineXCors, pLineYCors, pLineSID[numOfLines],
														  newOffsetS, offsetInLineArray, lineEquation);
					}
					coef1 = 1 / sqrt(1 + lineEquation[0] * lineEquation[0]);
					numOfOutlier = 0;
					newOffsetS = offsetInLineArray;
					while (offsetInEdgeArrayE > offsetInEdgeArrayS)
					{
						pointToLineDis = fabs(pEdgeXCors[offsetInEdgeArrayS] -
											  lineEquation[0] * pEdgeYCors[offsetInEdgeArrayS] - lineEquation[1]) * coef1;
						pLineXCors[offsetInLineArray] = pEdgeXCors[offsetInEdgeArrayS];
						pLineYCors[offsetInLineArray++] = pEdgeYCors[offsetInEdgeArrayS++];
						if (pointToLineDis > lineFitErrThreshold_)
						{
							numOfOutlier++;
							if (numOfOutlier > 3) break;
						}
						else  //we count number of connective outliers.
						{
							numOfOutlier = 0;
						}
					}
					//pop back the last few outliers from lines and return them to edge chain
					offsetInLineArray  -= numOfOutlier;
					offsetInEdgeArrayS -= numOfOutlier;
					if (offsetInLineArray - newOffsetS > 0 && tryTimes < TryTime) //some new pixels are added to the line
					{
					}
					else
					{
						bExtended = false;//no new pixels are added.
					}
				}
				//the line equation coefficients,for line w1x+w2y+w3 =0, we normalize it to make w1^2+w2^2 = 1.
				std::array<double, 3> lineEqu {1 * coef1, -lineEquation[0]*coef1, -lineEquation[1]*coef1};
				if (LineValidation_(pLineXCors, pLineYCors, pLineSID[numOfLines], offsetInLineArray, lineEqu, direction)) //check the line
				{
					//store the line equation coefficients
					lineEquations_.push_back(lineEqu);
					/*At last, compute the line endpoints and store them.
					 *we project the first and last pixels in the pixelChain onto the best fit line
					 *to get the line endpoints.
					 *xp= (w2^2*x0-w1*w2*y0-w3*w1)/(w1^2+w2^2)
					 *yp= (w1^2*y0-w1*w2*x0-w3*w2)/(w1^2+w2^2)  */
					std::array<float, 4> lineEndP;//line endpoints
					double a1 = lineEqu[1] * lineEqu[1];
					double a2 = lineEqu[0] * lineEqu[0];
					double a3 = lineEqu[0] * lineEqu[1];
					double a4 = lineEqu[2] * lineEqu[0];
					double a5 = lineEqu[2] * lineEqu[1];
					unsigned int Px = pLineXCors[pLineSID[numOfLines] ];//first pixel
					unsigned int Py = pLineYCors[pLineSID[numOfLines] ];
					lineEndP[0] = a1 * Px - a3 * Py - a4; //x
					lineEndP[1] = a2 * Py - a3 * Px - a5; //y
					Px = pLineXCors[offsetInLineArray - 1 ]; //last pixel
					Py = pLineYCors[offsetInLineArray - 1 ];
					lineEndP[2] = a1 * Px - a3 * Py - a4; //x
					lineEndP[3] = a2 * Py - a3 * Px - a5; //y
					lineEndpoints_.push_back(lineEndP);
					lineDirection_.push_back(direction);
					numOfLines++;
				}
				else
				{
					offsetInLineArray = pLineSID[numOfLines];// line was not accepted, the offset is set back
				}
			}
			//Extract line segments from the remaining pixel; Current chain has been shortened already.
		}
	}//end for(unsigned int edgeID=0; edgeID<edges.numOfEdges; edgeID++)

	pLineSID[numOfLines] = offsetInLineArray;
	lines.numOfLines = numOfLines;

#ifdef DEBUG_EDLINE
	if(!DEBUG_FOLDER.empty())
	{
		std::string line_folder = DEBUG_FOLDER+"/line/";
		int status = mkdir(line_folder.c_str(), 0777);

		if(!status)
		{
			/*Show the extracted lines in color. Each line is in different color.*/
			cout << "Line Drawing, numofLines = " << numOfLines << endl;

			cv::Mat img_all, img_single;
			cv::convertScaleAbs(image, img_all, double(UINT8_MAX)/UINT16_MAX);
			cv::cvtColor(img_all, img_all, CV_GRAY2BGR);
			img_single = img_all.clone();
			cv::Vec3b rColor;
			int lowest = 100, highest = 255;
			int range = (highest - lowest) + 1;

			srand((unsigned)time(0));

			for (unsigned int i = 0; i < lines.numOfLines; i++)
			{
				cv::Mat foo = img_single.clone();
				std::ofstream f(line_folder+std::to_string(i)+".txt");

				rColor.val[0] = lowest + int(rand() % range);
				rColor.val[1] = lowest + int(rand() % range);
				rColor.val[2] = lowest + int(rand() % range);

				for(offsetInLineArray = pLineSID[i]; offsetInLineArray<pLineSID[i+1]; offsetInLineArray++)
				{
					img_all.at<cv::Vec3b>(pLineYCors[offsetInLineArray], pLineXCors[offsetInLineArray])	= rColor;
					foo.at<cv::Vec3b>(pLineYCors[offsetInLineArray], pLineXCors[offsetInLineArray])	= cv::Vec3b{0x00,0x00,0xff};
					f << pLineYCors[offsetInLineArray] << " , " << pLineXCors[offsetInLineArray] << std::endl;
				}

				cv::imwrite(line_folder+std::to_string(i)+".png",foo);
				f.close();
			}
			cv::imwrite(DEBUG_FOLDER+"/lines.png", img_all);
		}
		else
		{
			std::cerr << "Fail to create line folder: " << line_folder << std::endl
					  << "Error No: " << errno << std::endl
					  << strerror(errno) << std::endl;
		}

	}

#endif
	return 1;
}

int EDLineDetector::EDline(cv::Mat const &image, bool smoothed)
{
	if ( EDline(image, lines_, smoothed) < 0 )
	{
		return -1;
	}

	lineSalience_.clear();
	lineSalience_.resize(lines_.numOfLines);
	unsigned char *pgImg  = gImgWO_.ptr();
	unsigned int indexInLineArray;
	unsigned int *pXCor = lines_.xCors.data();
	unsigned int *pYCor = lines_.yCors.data();
	unsigned int *pSID  = lines_.sId.data();
	for (unsigned int i = 0; i < lineSalience_.size(); i++)
	{
		int salience = 0;
		for (indexInLineArray = pSID[i]; indexInLineArray < pSID[i + 1]; indexInLineArray++)
		{
			salience += pgImg[pYCor[indexInLineArray] * imageWidth + pXCor[indexInLineArray] ];
		}
		lineSalience_[i] = (float) salience;
	}
	return 1;
}


double EDLineDetector::LeastSquaresLineFit_(  unsigned int *xCors,   unsigned int *yCors,
		unsigned int offsetS, std::array<double, 2> &lineEquation)
{
	if (lineEquation.size() != 2)
	{
		std::cout << "SHOULD NOT BE != 2" << std::endl;
		CV_Assert(false);
	}

	float * pMatT;
	float * pATA;
	double fitError = 0;
	double coef;
	unsigned char *pdirImg = dirImg_.data;
	unsigned int offset = offsetS;
	/*If the first pixel in this chain is horizontal,
	 *then we try to find a horizontal line, y=ax+b;*/
	if ( pdirImg[ yCors[offsetS]*imageWidth + xCors[offsetS] ] == Horizontal )
	{
		/*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [x0,1]         [y0]
		 * [x1,1] [a]     [y1]
		 *    .   [b]  =   .
		 * [xn,1]         [yn]*/
		pMatT = fitMatT.ptr<float>(); //fitMatT = [x0, x1, ... xn; 1,1,...,1];
		for (int i = 0; i < minLineLen_; i++)
		{
			//*(pMatT+minLineLen_) = 1; //the value are not changed;
			*(pMatT++)   = xCors[offsetS];
			fitVec[0][i] = yCors[offsetS++];
		}
		ATA = fitMatT * fitMatT.t();
		ATV = fitMatT * fitVec.t();
		/* [a,b]^T = Inv(mat^T * mat) * mat^T * vec */
		pATA = ATA.ptr<float>();
		coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
		// lineEquation = svd.Invert(ATA) * matT * vec;
		lineEquation[0] = coef * ( double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
		lineEquation[1] = coef * ( double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));
		/*compute line fit error */
		for (int i = 0; i < minLineLen_; i++)
		{
			coef = double(yCors[offset]) - double(xCors[offset++]) * lineEquation[0] - lineEquation[1];
			fitError += coef * coef;
		}
		return sqrt(fitError);
	}
	/*If the first pixel in this chain is vertical,
	 *then we try to find a vertical line, x=ay+b;*/
//	if (pdirImg[yCors[offsetS]*imageWidth + xCors[offsetS]] == Vertical)
	else
	{
		/*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [y0,1]         [x0]
		 * [y1,1] [a]     [x1]
		 *    .   [b]  =   .
		 * [yn,1]         [xn]*/
		pMatT = fitMatT.ptr<float>();//fitMatT = [y0, y1, ... yn; 1,1,...,1];
		for (int i = 0; i < minLineLen_; i++)
		{
			//*(pMatT+minLineLen_) = 1;//the value are not changed;
			*(pMatT++)           = yCors[offsetS];
			fitVec[0][i]          = xCors[offsetS++];
		}
		ATA = fitMatT * (fitMatT.t());
		ATV = fitMatT * fitVec.t();
		/* [a,b]^T = Inv(mat^T * mat) * mat^T * vec */
		pATA = ATA.ptr<float>();
		coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
		//      lineEquation = svd.Invert(ATA) * matT * vec;
		lineEquation[0] = coef * ( double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
		lineEquation[1] = coef * ( double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));
		/*compute line fit error */
		for (int i = 0; i < minLineLen_; i++)
		{
			coef = double(xCors[offset]) - double(yCors[offset++]) * lineEquation[0] - lineEquation[1];
			fitError += coef * coef;
		}
		return sqrt(fitError);
	}
	return 0;
}

double EDLineDetector::LeastSquaresLineFit_(  unsigned int *xCors,   unsigned int *yCors,
		unsigned int offsetS, unsigned int newOffsetS,
		unsigned int offsetE,   std::array<double, 2> &lineEquation)
{
	int length = offsetE - offsetS;
	int newLength = offsetE - newOffsetS;
	if (length <= 0 || newLength <= 0)
	{
		std::cout << "EDLineDetector::LeastSquaresLineFit_ Error:"
			 " the expected line index is wrong...offsetE = "
			 << offsetE << ", offsetS=" << offsetS << ", newOffsetS=" << newOffsetS << std::endl;
		return -1;
	}
	if (lineEquation.size() != 2)
	{
		std::cout << "SHOULD NOT BE != 2" << std::endl;
		CV_Assert(false);
	}
	cv::Mat_<float>  matT(2, newLength);
	cv::Mat_<float>  vec(newLength, 1);
	float * pMatT;
	float * pATA;
	//  double fitError = 0;
	double coef;
	unsigned char *pdirImg = dirImg_.data;
	/*If the first pixel in this chain is horizontal,
	 *then we try to find a horizontal line, y=ax+b;*/
	if (pdirImg[yCors[offsetS]*imageWidth + xCors[offsetS] ] == Horizontal)
	{
		/*Build the new system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [x0',1]         [y0']
		 * [x1',1] [a]     [y1']
		 *    .    [b]  =   .
		 * [xn',1]         [yn']*/
		pMatT = matT.ptr<float>();//matT = [x0', x1', ... xn'; 1,1,...,1]
		for (int i = 0; i < newLength; i++)
		{
			*(pMatT + newLength) = 1;
			*(pMatT++)         = xCors[newOffsetS];
			vec[0][i]          = yCors[newOffsetS++];
		}
		/* [a,b]^T = Inv(ATA + mat^T * mat) * (ATV + mat^T * vec) */
		tempMatLineFit = matT * matT.t();
		tempVecLineFit = matT * vec;
		ATA = ATA + tempMatLineFit;
		ATV = ATV + tempVecLineFit;
		pATA = ATA.ptr<float>();
		coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
		lineEquation[0] = coef * ( double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
		lineEquation[1] = coef * ( double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));
		/*compute line fit error */
		//      for(int i=0; i<length; i++){
		//          coef = double(yCors[offsetS]) - double(xCors[offsetS++]) * lineEquation[0] - lineEquation[1];
		//          fitError += coef*coef;
		//      }
		return 0;
	}
	/*If the first pixel in this chain is vertical,
	 *then we try to find a vertical line, x=ay+b;*/
	if (pdirImg[yCors[offsetS]*imageWidth + xCors[offsetS] ] == Vertical)
	{
		/*Build the system,and solve it using least square regression: mat * [a,b]^T = vec
		 * [y0',1]         [x0']
		 * [y1',1] [a]     [x1']
		 *    .    [b]  =   .
		 * [yn',1]         [xn']*/
//		pMatT = matT.GetData();//matT = [y0', y1', ... yn'; 1,1,...,1]
		pMatT = matT.ptr<float>();//matT = [y0', y1', ... yn'; 1,1,...,1]
		for (int i = 0; i < newLength; i++)
		{
			*(pMatT + newLength) = 1;
			*(pMatT++)         = yCors[newOffsetS];
			vec[0][i]             = xCors[newOffsetS++];
		}
		/* [a,b]^T = Inv(ATA + mat^T * mat) * (ATV + mat^T * vec) */
//		matT.MultiplyWithTransposeOf(matT, tempMatLineFit);
		tempMatLineFit = matT * matT.t();
		tempVecLineFit = matT * vec;
		ATA = ATA + tempMatLineFit;
		ATV = ATV + tempVecLineFit;
//		pATA = ATA.GetData();
		pATA = ATA.ptr<float>();
		coef = 1.0 / (double(pATA[0]) * double(pATA[3]) - double(pATA[1]) * double(pATA[2]));
		lineEquation[0] = coef * ( double(pATA[3]) * double(ATV[0][0]) - double(pATA[1]) * double(ATV[0][1]));
		lineEquation[1] = coef * ( double(pATA[0]) * double(ATV[0][1]) - double(pATA[2]) * double(ATV[0][0]));
		/*compute line fit error */
		//      for(int i=0; i<length; i++){
		//          coef = double(xCors[offsetS]) - double(yCors[offsetS++]) * lineEquation[0] - lineEquation[1];
		//          fitError += coef*coef;
		//      }
	}
	return 0;
}

bool EDLineDetector::LineValidation_(  unsigned int *xCors,   unsigned int *yCors,
									   unsigned int offsetS, unsigned int offsetE,
									   std::array<double, 3> &lineEquation, float &direction)
{
	if (bValidate_)
	{
		int n = offsetE - offsetS;
		/*first compute the direction of line, make sure that the dark side always be the
		 *left side of a line.*/
		int meanGradientX = 0, meanGradientY = 0;
		float *pdxImg = dxImg_.ptr<float>();
		float *pdyImg = dyImg_.ptr<float>();
		double dx, dy;
		std::vector<double> pointDirection;
		int index;
		for (int i = 0; i < n; i++)
		{
			index = yCors[offsetS] * imageWidth + xCors[offsetS++];
			meanGradientX += pdxImg[index];
			meanGradientY += pdyImg[index];
			dx = (double) pdxImg[index];
			dy = (double) pdyImg[index];
			pointDirection.push_back(atan2(-dx, dy));
		}
		dx = fabs(lineEquation[1]);
		dy = fabs(lineEquation[0]);
		if (meanGradientX == 0 && meanGradientY == 0) //not possible, if happens, it must be a wrong line,
		{
			return false;
		}
		if (meanGradientX > 0 && meanGradientY >= 0) //first quadrant, and positive direction of X axis.
		{
			direction = atan2(-dy, dx); //line direction is in fourth quadrant
		}
		if (meanGradientX <= 0 && meanGradientY > 0) //second quadrant, and positive direction of Y axis.
		{
			direction  = atan2(dy, dx); //line direction is in first quadrant
		}
		if (meanGradientX < 0 && meanGradientY <= 0) //third quadrant, and negative direction of X axis.
		{
			direction = atan2(dy, -dx); //line direction is in second quadrant
		}
		if (meanGradientX >= 0 && meanGradientY < 0) //fourth quadrant, and negative direction of Y axis.
		{
			direction = atan2(-dy, -dx); //line direction is in third quadrant
		}
		/*then check whether the line is on the border of the image. We don't keep the border line.*/
		if (fabs(direction) < 0.15 || M_PI - fabs(direction) < 0.15) //Horizontal line
		{
			if (fabs(lineEquation[2]) < 10 || fabs(imageHeight - fabs(lineEquation[2])) < 10) //upper border or lower border
			{
				return false;
			}
		}
		if (fabs(fabs(direction) - M_PI * 0.5) < 0.15) //Vertical line
		{
			if (fabs(lineEquation[2]) < 10 || fabs(imageWidth - fabs(lineEquation[2])) < 10) //left border or right border
			{
				return false;
			}
		}
		//count the aligned points on the line which have the same direction as the line.
		double disDirection;
		int k = 0;
		for (int i = 0; i < n; i++)
		{
			disDirection = fabs(direction - pointDirection[i]);
			if (fabs(2 * M_PI - disDirection) < 0.392699 || disDirection < 0.392699) //same direction, pi/8 = 0.392699081698724
			{
				k++;
			}
		}
		//now compute NFA(Number of False Alarms)
		double ret = nfa(n, k, 0.125, logNT_);

		return (ret > 0); //0 corresponds to 1 mean false alarm
	}
	else
	{
		return true;
	}
}
