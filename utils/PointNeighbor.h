#pragma once

#include "Point.h"
#include "opencv2/opencv.hpp"

const Point3D Neighbour_26[26] =
{
	Point3D(-1,0,0),
	Point3D(1,0,0),
	Point3D(0,-1,0),
	Point3D(0,1,0),
	Point3D(0,0,-1),
	Point3D(0,0,1),

	Point3D(-1,0,-1),
	Point3D(1,0,-1),
	Point3D(0,-1,0),
	Point3D(0,1,-1),
	Point3D(-1,0,1),
	Point3D(1,0,1),
	Point3D(0,-1,1),
	Point3D(0,1,1),
	Point3D(-1,-1,0),
	Point3D(-1,1,0),
	Point3D(1,-1,0),
	Point3D(1,1,0),

	Point3D(-1,-1,-1),
	Point3D(-1,1,-1),
	Point3D(1,-1,-1),
	Point3D(-1,-1,1),
	Point3D(-1,1,1),
	Point3D(1,1,1)
};

const Point3D Neighbour_6[6] =
{
	Point3D(-1,0,0),
	Point3D(0,-1,0),
	Point3D(0,0,-1),
	Point3D(0,0,1),
	Point3D(0,1,0),
	Point3D(1,0,0),
};


const Point2D Neighbour2D_4[4] =
{
	Point2D(-1,0),
	Point2D(1,0),
	Point2D(0,-1),
	Point2D(0,1)
};


const Point2D Neighbour2D_8[8] =
{
	Point2D(-1,0),
	Point2D(1,0),
	Point2D(0,-1),
	Point2D(0,1),

	Point2D(-1,-1),
	Point2D(-1,1),
	Point2D(1,-1),
	Point2D(1,1)
};


const cv::Point2f CvNeighbour2D_8[8] =
{
	cv::Point2f(-1,0),
	cv::Point2f(1,0),
	cv::Point2f(0,-1),
	cv::Point2f(0,1),

	cv::Point2f(-1,-1),
	cv::Point2f(-1,1),
	cv::Point2f(1,-1),
	cv::Point2f(1,1)
};
