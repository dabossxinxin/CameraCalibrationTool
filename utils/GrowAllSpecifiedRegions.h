// 生长出所有符合自定义条件的区域
#pragma once

#include "Point.h"
#include "PointNeighbor.h"
#include <opencv2/opencv.hpp>
#include <limits>
#include <vector>

class MtxGrowAllSpecifiedRegions
{
public:
	MtxGrowAllSpecifiedRegions();

	~MtxGrowAllSpecifiedRegions();

	void SetInputData(cv::Mat pData);
	void SetInputData(unsigned char *pData, int w, int h);

	void SetRoiMask(unsigned char *);

	void SetRoiExtent(int *extent) {
		m_pExtent = extent;
	};

	void SetSearchStep(int step) {
		m_iStep = step;
	};

	void SetResizeFlag(bool flag) {
		m_bFlag = flag;
	}

	void SetResizeScale(float scale) {
		m_fScale = scale;
	}

	inline void SetUpperLimitVal(int val)
	{
		m_iUpperLimit = val;
	}

	inline void SetLowerLimitVal(int val)
	{
		m_iLowerLimit = val;
	}

	void SetAndOr(int type) {
		m_iAndOr = type;
	};

	inline std::vector<std::vector<int>> GetSpecifiedRegions()
	{
		return m_TargetRegions;
	}

	bool Update();


protected:
	void run();

private:
	//template<class T>
	//bool SpecifiedRegionsGrow(T *pData);

	template<class T>
	bool SpecifiedRegionsGrow(T *pData, int step = 1);

	void ClearQueue(std::queue<Point2D>& q);

	void RefineScale();

	bool					m_bRunState;

	unsigned char			*m_pInputData;
	unsigned char			*m_pRoiMask;

	int						m_iWidth;
	int						m_iHeight;

	int						m_iWidthScaled;
	int						m_iHeightScaled;

	int						m_iUpperLimit;
	int						m_iLowerLimit;

	int						*m_pExtent;

	std::vector<std::vector<int>>	m_TargetRegions;

	int						m_iStep;
	
	bool					m_bFlag;
	
	float					m_fScale;

	int						m_iAndOr;
};
