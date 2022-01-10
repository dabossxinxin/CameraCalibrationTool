#include "GrowAllSpecifiedRegions.h"

//#define SAVE_INFO

MtxGrowAllSpecifiedRegions::MtxGrowAllSpecifiedRegions() : m_iStep(1), m_pExtent(NULL),
m_bRunState(false), m_pInputData(NULL), m_pRoiMask(NULL), m_iWidth(0), m_iHeight(0),m_iAndOr(1)
{
	m_iUpperLimit = 255;
	m_iLowerLimit = 128;

	m_bFlag = false;
	m_fScale = 1.0;

	m_TargetRegions.clear();
}

MtxGrowAllSpecifiedRegions::~MtxGrowAllSpecifiedRegions()
{

}

void MtxGrowAllSpecifiedRegions::SetInputData(cv::Mat pData)
{
	m_iHeight = pData.rows;
	m_iWidth = pData.cols;

	m_pInputData = (unsigned char *)malloc(sizeof(char)*m_iWidth*m_iHeight);
	int index = 0;
	for (int i = 0; i < m_iHeight; i++)
	{
		for (int j = 0; j < m_iWidth; j++)
		{
			m_pInputData[index] = pData.at<unsigned char>(i, j);
			index++;
		}
	}
}

void MtxGrowAllSpecifiedRegions::SetInputData(unsigned char *pData, int w, int h)
{
	if (nullptr == pData)
		return;

	if (m_bFlag)
	{
		cv::Mat ImageData = cv::Mat(h, w, CV_8U, pData, 0);
		cv::Size sz(cvRound((float)w*m_fScale), cvRound((float)h*m_fScale));
		cv::resize(ImageData, ImageData, sz, 0, 0, cv::INTER_LINEAR);
		m_pInputData = (unsigned char*)malloc(sz.height*sz.width);
		memcpy(m_pInputData, ImageData.data, sz.width*sz.height);
		m_iWidth = w;
		m_iHeight = h;
		m_iWidthScaled = cvRound((float)w*m_fScale);
		m_iHeightScaled = cvRound((float)h*m_fScale);

#ifdef SAVE_INFO
		std::string fileResize = "F:\\Users\\Admin\\Desktop\\resize.jpg";
		cv::imwrite(fileResize, ImageData);
#endif
	}
	else
	{
		m_iWidth = w;
		m_iHeight = h;
		//m_iWidthScaled = cvRound((float)w*m_fScale);
		//m_iHeightScaled = cvRound((float)h*m_fScale);
		m_pInputData = pData;
	}
}

void MtxGrowAllSpecifiedRegions::SetRoiMask(unsigned char *pData)
{
	m_pRoiMask = pData;
}

bool MtxGrowAllSpecifiedRegions::Update()
{
	this->run();
	return m_bRunState;
}

void MtxGrowAllSpecifiedRegions::run()
{
	if (m_pInputData == NULL)
	{
		m_bRunState = false;
		return;
	}

	int extent[4];
	if (NULL == m_pExtent)
	{
		if (m_bFlag)
		{
			m_pExtent = extent;
			m_pExtent[0] = 0;
			m_pExtent[1] = m_iWidthScaled - 1;
			m_pExtent[2] = m_iHeightScaled - 1;
			m_pExtent[3] = 0;
		}
		else
		{
			m_pExtent = extent;
			m_pExtent[0] = 0;
			m_pExtent[1] = m_iWidth - 1;
			m_pExtent[2] = m_iHeight - 1;
			m_pExtent[3] = 0;
		}
	}
	else
	{
		if (m_bFlag)
		{
			m_pExtent[0] *= m_fScale;
			m_pExtent[1] *= m_fScale;
			m_pExtent[2] *= m_fScale;
			m_pExtent[3] *= m_fScale;
		}
	}

	int needFreeRoiMask = 0;
	if (m_pRoiMask == nullptr)
	{
		int imgSize2D = m_iWidth * m_iHeight;
		if (m_bFlag) imgSize2D = m_iWidthScaled * m_iHeightScaled;
		
		m_pRoiMask = new unsigned char[imgSize2D];
		memset(m_pRoiMask, 1, sizeof(unsigned char) * imgSize2D);
		needFreeRoiMask = 1;
	}
	else
	{
		if (m_bFlag)
		{
			cv::Mat Roi = cv::Mat(m_iHeight, m_iWidth, CV_8U, m_pRoiMask, 0);
			cv::resize(Roi, Roi, cv::Size(m_iWidthScaled, m_iHeightScaled), 0, 0, cv::INTER_LINEAR);
			m_pRoiMask = (unsigned char*)malloc(m_iWidthScaled*m_iHeightScaled);
			memcpy(m_pRoiMask, Roi.data, m_iWidthScaled*m_iHeightScaled);
			needFreeRoiMask = 1;
		}
	}

	m_bRunState = this->SpecifiedRegionsGrow(m_pInputData, m_iStep);

	if (needFreeRoiMask)
	{
		free(m_pRoiMask);
		m_pRoiMask = nullptr;
	}
}

void MtxGrowAllSpecifiedRegions::RefineScale()
{
	if (m_bFlag)
	{
		const float scaleInv = 1.0 / m_fScale;
		const int RegionsNum = m_TargetRegions.size();
		for (int it = 0; it < RegionsNum; ++it)
		{
			const int PixelsNum = m_TargetRegions[it].size();
			for (int i = 0; i < PixelsNum; ++i)
			{
				const int row = m_TargetRegions[it][i] / m_iWidthScaled;
				const int col = m_TargetRegions[it][i] % m_iWidthScaled;
				const int index = row*scaleInv*m_iWidth + col*scaleInv;
				m_TargetRegions[it][i] = index;
			}
		}
	}
}

template<class T>
bool MtxGrowAllSpecifiedRegions::SpecifiedRegionsGrow(T *pData, int step)
{
	if (nullptr == pData)
		return false;

	int width = m_iWidth;
	int height = m_iHeight;
	if (m_bFlag)
	{
		width = m_iWidthScaled;
		height = m_iHeightScaled;
	}

	int imgSize2D = width * height;
	std::vector<Point2D> vec;
	std::vector<int> region;
	m_TargetRegions.clear();

	char* flag = new char[imgSize2D];
	memset(flag, 0, sizeof(char)*imgSize2D);

	Point2D tempPt;

	int index = 0;

	if (m_iAndOr)
	{
		for (int i = m_pExtent[3]; i <= m_pExtent[2]; i++)
		{
			int index = i * width + m_pExtent[0];
			for (int j = m_pExtent[0]; j <= m_pExtent[1]; j++)
			{
				if (pData[index] >= m_iLowerLimit && pData[index] <= m_iUpperLimit
					&& m_pRoiMask[index] != 0 && flag[index] == 0) //规则调整
				{
					Point2D seed(j, i);
					flag[index] = 1;
					vec.clear();
					vec.push_back(seed);
					region.clear();
					region.push_back(index);

					while (!vec.empty())
					{
						tempPt = vec.back();
						vec.pop_back();
						for (int k = 0; k < 8; k++)
						{
							Point2D newPoint = tempPt + Neighbour2D_8[k];

							if (newPoint.Y >= m_pExtent[3] && newPoint.Y <= m_pExtent[2] &&
								newPoint.X >= m_pExtent[0] && newPoint.X <= m_pExtent[1])
							{
								int newIndex = newPoint.Y*width + newPoint.X;
								if (flag[newIndex] == 0 && m_pRoiMask[newIndex] != 0 &&
									pData[newIndex] >= m_iLowerLimit && pData[newIndex] <= m_iUpperLimit)  //规则调整
								{
									vec.push_back(newPoint);
									region.push_back(newIndex);
									flag[newIndex] = 1;
								}
							}

						}
					}

					m_TargetRegions.push_back(region);
				}

				index++;
			}
		}
	}
	else
	{
		for (int i = m_pExtent[3]; i <= m_pExtent[2]; i++)
		{
			int index = i * width + m_pExtent[0];
			for (int j = m_pExtent[0]; j <= m_pExtent[1]; j++)
			{
				if ((pData[index] >= m_iLowerLimit || pData[index] <= m_iUpperLimit)
					&& m_pRoiMask[index] != 0 && flag[index] == 0) //规则调整
				{
					Point2D seed(j, i);
					flag[index] = 1;
					vec.clear();
					vec.push_back(seed);
					region.clear();
					region.push_back(index);

					while (!vec.empty())
					{
						tempPt = vec.back();
						vec.pop_back();
						for (int k = 0; k < 8; k++)
						{
							Point2D newPoint = tempPt + Neighbour2D_8[k];

							if (newPoint.Y >= m_pExtent[3] && newPoint.Y <= m_pExtent[2] &&
								newPoint.X >= m_pExtent[0] && newPoint.X <= m_pExtent[1])
							{
								int newIndex = newPoint.Y*width + newPoint.X;
								if (flag[newIndex] == 0 && m_pRoiMask[newIndex] != 0 &&
									(pData[newIndex] >= m_iLowerLimit || pData[newIndex] <= m_iUpperLimit))  //规则调整
								{
									vec.push_back(newPoint);
									region.push_back(newIndex);
									flag[newIndex] = 1;
								}
							}
						}
					}

					m_TargetRegions.push_back(region);
				}

				index++;
			}
		}
	}
	
	this->RefineScale();
	delete[] flag; flag = nullptr;

	return true;
}



void MtxGrowAllSpecifiedRegions::ClearQueue(std::queue<Point2D>& q)
{
	std::queue<Point2D> empty;
	swap(empty, q);
}
