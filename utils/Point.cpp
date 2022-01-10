#include "Point.h"

Point3D::Point3D() : X(0), Y(0), Z(0) {};

Point3D::Point3D(int x, int y, int z) 
{
	X = x;
	Y = y;
	Z = z;
}

Point3D::Point3D(const Point3D &p) 
{
	X = p.X;
	Y = p.Y;
	Z = p.Z;
}

bool Point3D::operator==(const Point3D &p) const 
{
	return (p.X == X && p.Y == Y && p.Z == Z);
}

Point3D Point3D::operator+(const Point3D &p) const 
{
	return Point3D(X + p.X, Y + p.Y, Z + p.Z);
}

Point3D Point3D::operator-(const Point3D &p) const 
{
	return Point3D(X - p.X, Y - p.Y, Z - p.Z);
}

Point3Dd::Point3Dd() : X(0), Y(0), Z(0) {};

Point3Dd::Point3Dd(double x, double y, double z)
{
	X = x;
	Y = y;
	Z = z;
}

Point3Dd::Point3Dd(const Point3Dd &p)
{
	X = p.X;
	Y = p.Y;
	Z = p.Z;
}

bool Point3Dd::operator==(const Point3Dd &p) const
{
	return (p.X == X && p.Y == Y && p.Z == Z);
}

Point3Dd Point3Dd::operator+(const Point3Dd &p) const
{
	return Point3Dd(X + p.X, Y + p.Y, Z + p.Z);
}

Point3Dd Point3Dd::operator-(const Point3Dd &p) const
{
	return Point3Dd(X - p.X, Y - p.Y, Z - p.Z);
}


// --------------------------------------------------------------------
Point2D::Point2D() : X(0), Y(0) {};

Point2D::Point2D(int x, int y)
{
	X = x;
	Y = y;
}

Point2D::Point2D(const Point2D &p)
{
	X = p.X;
	Y = p.Y;
}

bool Point2D::operator==(const Point2D &p) const
{
	return (p.X == X && p.Y == Y);
}

Point2D Point2D::operator+(const Point2D &p) const
{
	return Point2D(X + p.X, Y + p.Y);
}

Point2D Point2D::operator-(const Point2D &p) const
{
	return Point2D(X - p.X, Y - p.Y);
}

// --------------------------------------------------------------------
Point2Dd::Point2Dd() : X(0), Y(0) {};

Point2Dd::Point2Dd(double x, double y)
{
	X = x;
	Y = y;
}

Point2Dd::Point2Dd(const Point2Dd &p)
{
	X = p.X;
	Y = p.Y;
}

bool Point2Dd::operator==(const Point2Dd &p) const
{
	return (p.X == X && p.Y == Y);
}

Point2Dd Point2Dd::operator+(const Point2Dd &p) const
{
	return Point2Dd(X + p.X, Y + p.Y);
}

Point2Dd Point2Dd::operator-(const Point2Dd &p) const
{
	return Point2Dd(X - p.X, Y - p.Y);
}
