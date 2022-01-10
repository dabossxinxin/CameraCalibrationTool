#pragma once

class Point3D
{
public:
	Point3D();

	Point3D(int x, int y, int z);

	Point3D(const Point3D &p);

	bool operator==(const Point3D &p) const;

	Point3D operator+(const Point3D &p) const;

	Point3D operator-(const Point3D &p) const;

	int X;
	int Y;
	int Z;
};


class Point3Dd
{
public:
	Point3Dd();

	Point3Dd(double x, double y, double z);

	Point3Dd(const Point3Dd &p);

	bool operator==(const Point3Dd &p) const;

	Point3Dd operator+(const Point3Dd &p) const;

	Point3Dd operator-(const Point3Dd &p) const;

	double X;
	double Y;
	double Z;
};


class Point2D
{
public:
	Point2D();

	Point2D(int x, int y);

	Point2D(const Point2D &p);

	bool operator==(const Point2D &p) const;

	Point2D operator+(const Point2D &p) const;

	Point2D operator-(const Point2D &p) const;

	int X;
	int Y;
};

class Point2Dd
{
public:
	Point2Dd();

	Point2Dd(double x, double y);

	Point2Dd(const Point2Dd &p);

	bool operator==(const Point2Dd &p) const;

	Point2Dd operator+(const Point2Dd &p) const;

	Point2Dd operator-(const Point2Dd &p) const;

	double X;
	double Y;
};


