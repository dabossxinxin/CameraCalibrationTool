#include "RegionDetect.h"



namespace MtRegionDetect
{
	//this Function compute the maxmum between-class variance
	int RegionDetect::Otsu(const cv::Mat& image)
	{
		int th;
		const int GrayScale = 256;					// Total Gray level of gray image
		int pixCount[GrayScale] = { 0 };			// The number of pixels occupied by each gray value
		int pixSum = image.rows * image.cols;		// The total number of pixels of the image
		float pixProportion[GrayScale] = { 0.0 };	// The proportion of total pixels occupied by each gray value

		float w0, w1, u0tmp, u1tmp, u0, u1, deltaTmp, deltaMax = 0;

		//std::ofstream out("E:\\Code\\EllipseFitSource\\data\\pixel.txt");

		// Count the number of pixels in each gray level
		for (int i = 0; i < image.cols; i++)
		{
			for (int j = 0; j < image.rows; j++)
			{
				int size = image.at<uchar>(j, i);
				pixCount[image.at<uchar>(j, i)]++;
				//out << size << " ";
			}
		}
		//out.close();

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

	void RegionDetect::Initialize()
	{
		dx.resize(4);
		dy.resize(4);

		dx[0] = -1;;
		dx[1] = 0;
		dx[2] = 1;
		dx[3] = 0;

		dy[0] = 0;
		dy[1] = 1;
		dy[2] = 0;
		dy[3] = -1;

		int Row = mGrayImage.rows;
		int Col = mGrayImage.cols;
		mIsVisited.resize(Row);
		for (int it = 0; it < Row; it++)
			mIsVisited[it].resize(Col);

		for (int i = 0; i < Row; i++)
		{
			for (int j = 0; j < Col; j++)
			{
				mIsVisited[i][j] = false;
			}
		}
	}
	
	void RegionDetect::InitialImage(cv::Mat& image, const int& value)
	{
		const int Row = image.rows;
		const int Col = image.cols;
		
		for (int row = 0; row < Row; row++)
		{
			for (int col = 0; col < Col; col++)
			{
				image.at<uchar>(row, col) = value;
			}
		}
	}

	void RegionDetect::BFS_Iteration(const cv::Mat& input, const int& th, int& count, std::vector<cv::Mat>& output)throw(RegionDetectFailure)
	{
		count = 0;
		std::queue<int> que;
		const int Row = input.rows;
		const int Col = input.cols;

		if (Row == 0 || Col == 0)
		{
			throw RegionDetectFailure("The Input Image Is Empty!");
		}

		for (int row = 0; row < Row; row++)
		{
			for (int col = 0; col < Col; col++)
			{
				if (input.at<uchar>(row, col) >= th && mIsVisited[row][col] == false)
				{
					que.push(row);
					que.push(col);

					mIsVisited[row][col] = true;

					cv::Mat out(mImage.rows, mImage.cols, mImage.type());
					InitialImage(out, 0);
					
					out.at<uchar>(row, col) = input.at<uchar>(row, col);
					while (!que.empty())
					{
						int x = que.front();
						que.pop();
						int y = que.front();
						que.pop();
						for (int it = 0; it < 4; it++)
						{
							int xx = x + dx[it];
							int yy = y + dy[it];
							if (xx < 0 || xx >= Row || yy < 0 || yy >= Col)
							{
								continue;
							}
							if (input.at<uchar>(xx, yy) >= th && mIsVisited[xx][yy] == false)
							{
								out.at<uchar>(xx, yy) = input.at<uchar>(xx, yy);
								mIsVisited[xx][yy] = true;
								que.push(xx);
								que.push(yy);
							}
						}
					}
					count++;
					output.push_back(out);
				}
			}
		}
	}

	void RegionDetect::BFS_Recursion(const cv::Mat& input, const int& row , const int& col, const int& th, cv::Mat& output)
	{
		if (row < 0 || row >= input.rows || col < 0 || col >= output.cols ||
			input.at<uchar>(row, col) < th || mIsVisited[row][col] == true)
		{
			return;
		}

		mIsVisited[row][col] = true;
		output.at<uchar>(row, col) = input.at<uchar>(row, col);
		
		for (int it = 0; it < 4; it++)
		{
			int x = row + dx[it];
			int y = col + dy[it];
			BFS_Recursion(input, x, y, th, output);
		}
	}

	void RegionDetect::compute(MatrixP& region, const int& threshold)throw(RegionDetectFailure)
	{
		if (mImage.channels() == 3)
		{
			cv::cvtColor(mImage, mGrayImage, cv::COLOR_BGR2GRAY);
		}
		else if (mImage.channels() == 1)
		{
			mGrayImage = mImage.clone();
		}
		
		const int Row = mGrayImage.rows;
		const int Col = mGrayImage.cols;
		
		int count = RegionGrowth(mGrayImage, threshold, region);

		if (count <= 0)
		{
			throw RegionDetectFailure("Input Image Without Texture Feature!");
		}
		else
		{
			return;
		}
		
	}

	void RegionDetect::compute(MatrixMat& region, const int& threshold)throw(RegionDetectFailure)
	{
		if (mImage.channels() == 3)
		{
			cv::cvtColor(mImage, mGrayImage, cv::COLOR_BGR2GRAY);
		}
		else if (mImage.channels() == 1)
		{
			mGrayImage = mImage.clone();
		}

		const int Row = mGrayImage.rows;
		const int Col = mGrayImage.cols;
		
		cv::Mat tmp(Row,Col,mImage.type());

		int count = RegionGrowth(mGrayImage, threshold,region,BFSMethod::Iteration);

		if (count <= 0)
		{
			throw RegionDetectFailure("Input Image Without Texture Feature!");
		}
		else
		{
			return;
		}
	}

	int RegionDetect::RegionGrowth(const cv::Mat& image,const int& th, MatrixMat& region, const BFSMethod& method)throw(RegionDetectFailure)
	{
		int Row, Col;
		int count = 0;
		
		Row = image.rows;
		Col = image.cols;
		
		if (Row == 0 || Col == 0)
		{
			throw RegionDetectFailure("Input Image Error!");
			return count;
		}

		Initialize();

		if (method == BFSMethod::Recursion)
		{
			for (int i = 0; i < Row; i++)
			{
				for (int j = 0; j < Col; j++)
				{
					if (image.at<uchar>(i, j) >= th && mIsVisited[i][j] == false)
					{
						count++;

						cv::Mat output(mImage.rows, mImage.cols, mImage.type());
						InitialImage(output, 0);
						BFS_Recursion(image, i, j, th, output);
						region.push_back(output);
					}
				}
			}
		}
		else if (method == BFSMethod::Iteration)
		{
			std::vector<cv::Mat> output;
			BFS_Iteration(image, th, count, output);
			region = output;
		}
		return count;
	}

	void RegionDetect::BFS_Iteration(const cv::Mat& input, const int& th, int& count, std::vector<std::vector<pair<cv::Point2i,int>>>& output)throw(RegionDetectFailure)
	{
		count = 0;
		std::queue<int> que;
		const int Row = input.rows;
		const int Col = input.cols;

		if (Row == 0 || Col == 0)
		{
			throw RegionDetectFailure("The Input Image Is Empty!");
		}

		for (int row = 0; row < Row; row++)
		{
			for (int col = 0; col < Col; col++)
			{
				if (input.at<uchar>(row, col) >= th && mIsVisited[row][col] == false)
				{
					que.push(row);
					que.push(col);

					mIsVisited[row][col] = true;

					/*cv::Mat out(mImage.rows, mImage.cols, mImage.type());
					InitialImage(out, 0);*/
					
					std::vector<pair<cv::Point2i,int>> out;
				
					pair<cv::Point2i, int> p1;
					p1.first.x = col;
					p1.first.y = row;
					p1.second = input.at<uchar>(row, col);
					out.push_back(p1);

					while (!que.empty())
					{
						int x = que.front();
						que.pop();
						int y = que.front();
						que.pop();
						for (int it = 0; it < 4; it++)
						{
							int xx = x + dx[it];
							int yy = y + dy[it];
							if (xx < 0 || xx >= Row || yy < 0 || yy >= Col)
							{
								continue;
							}
							if (input.at<uchar>(xx, yy) >= th && mIsVisited[xx][yy] == false)
							{
								pair<cv::Point2f, int> p2;
								p2.first.x = yy;
								p2.first.y = xx;
								p2.second = input.at<uchar>(xx, yy);
								out.push_back(p2);
								/*out.at<uchar>(xx, yy) = input.at<uchar>(xx, yy);*/
								mIsVisited[xx][yy] = true;
								que.push(xx);
								que.push(yy);
							}
						}
					}
					count++;
					output.push_back(out);
				}
			}
		}
	}

	void getMinMax2D(const std::vector<pair<cv::Point2i,int>>& pixel, int& uMin, int& uMax, int& vMin, int& vMax)
	{
		const int number = pixel.size();

		uMin = INT_MAX;
		uMax = -INT_MAX;
		vMin = INT_MAX;
		vMax = -INT_MAX;

		for (int i = 0; i < number; i++)
		{
			if (pixel[i].first.x > uMax)
				uMax = pixel[i].first.x;
			if (pixel[i].first.x < uMin)
				uMin = pixel[i].first.x;
			if (pixel[i].first.y > vMax)
				vMax = pixel[i].first.y;
			if (pixel[i].first.y < vMin)
				vMin = pixel[i].first.y;
		}

		uMax += 1;
		vMax += 1;
		if (uMin > 0)
			uMin -= 1;
		if (vMin > 0)
			vMin -= 1;
	}

	int RegionDetect::RegionGrowth(const cv::Mat& image, const int& th, MatrixP& region)throw(RegionDetectFailure)
	{
		int Row, Col;
		int count = 0;

		Row = image.rows;
		Col = image.cols;

		if (Row == 0 || Col == 0)
		{
			throw RegionDetectFailure("Input Image Error!");
			return count;
		}

		Initialize();

		std::vector<std::vector<pair<cv::Point2i,int>>> output;
		BFS_Iteration(image, th, count,output);
		
		const int number = output.size();
		for (int i = 0; i < number; i++)
		{
			pair<cv::Mat, cv::Point2f> tmp;
			
			int uMax, uMin, vMax, vMin;
			getMinMax2D(output[i], uMin, uMax, vMin, vMax);

			// 向外扩张EdgePixel个像素
			const int row = vMax - vMin + EdgePixel;
			const int col = uMax - uMin + EdgePixel;
			
			cv::Mat out(row, col, mImage.type());
			InitialImage(out, 0);

			for (int j = 0; j < output[i].size(); j++)
			{
				out.at<uchar>(output[i][j].first.y-vMin+EdgePixel/2,output[i][j].first.x-uMin+EdgePixel/2)
					= output[i][j].second;
			}

			tmp.first = out;
			tmp.second.x = uMin - EdgePixel/2;
			tmp.second.y = vMin - EdgePixel/2;

			region.push_back(tmp);
		}

		return count;
	}

	RegionDetectFailure::RegionDetectFailure(const char* msg) :Message(msg) {};
	const char* RegionDetectFailure::GetMessage() { return Message; }
}