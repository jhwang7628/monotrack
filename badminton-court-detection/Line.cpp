//
// Created by Chlebus, Grzegorz on 25.08.17.
// Copyright (c) Chlebus, Grzegorz. All rights reserved.
//

#include "Line.h"
#include "geometry.h"

using namespace cv;

Line Line::fromRhoTheta(float rho, float theta)
{
  double a = cos(theta), b = sin(theta);
  double x0 = a * rho, y0 = b * rho;
  Point2f p1((x0 + 2000 * (-b)),
    (y0 + 2000 * (a)));
  Point2f p2((x0 - 2000 * (-b)),
    (y0 - 2000 * (a)));
  return Line::fromTwoPoints(p1, p2);
}


Line Line::fromTwoPoints(cv::Point2f p1, cv::Point2f p2)
{
  Point2f vec = p2 - p1;
  // Make directions always point positive so that we can
  // compare them reliably.
  if (vec.x < 0) {
    vec.x *= -1;
    vec.y *= -1;
  }
  return Line(p1, vec);
}

Line::Line(cv::Point2f point, cv::Point2f vector)
  : u(point)
  , v(normalize(vector))
{

}

Line::Line() 
  : u(cv::Point2f(0,0))
  , v(cv::Point2f(0,0))
{
  
}

cv::Point2f Line::getPoint() const
{
  return u;
}


cv::Point2f Line::getVector() const
{
  return v;
}

// This is a signed perpendicular distance
float Line::getPerpendicularDistance(const cv::Point2f& point) const
{
  Point2f pointOnLine = getPointOnLineClosestTo(point);
  return (point - pointOnLine).dot(perpendicular(v));
}

float Line::getDistance(const cv::Point2f& point) const
{
  Point2f pointOnLine = getPointOnLineClosestTo(point);
  return distance(point, pointOnLine);
}

cv::Point2f Line::getPointOnLineClosestTo(const cv::Point2f point) const
{
  // u + [(p-u) . v] * v
  return u + v.dot(point - u) * v;
}

bool Line::isDuplicate(const Line& otherLine) const
{
  // Check if (0, 0) is close to both lines. Two lines are close enough
  // for our applications if their slopes and intercepts are roughly equal
  auto zero = cv::Point2f(0, 0);
  double thresh = abs(otherLine.getDistance(zero) - getDistance(zero));
  if (thresh < PIXEL_EPS && distance(otherLine.v, v) < BIG_EPS) {
    return true;
  } else {
    return false;
  }
}

bool Line::isParallel(const Line& otherLine, double tol) const
{
  if (distance(otherLine.v, v) < tol) {
    return true;
  } else {
    return false;
  }
}

float Line::evaluateByX(float x) const {
  if (v.x == 0) return INFINITY;
  return u.y + (x - u.x) * v.y / v.x;
}

void Line::toImplicit(cv::Point2f& n, float& c) const
{
  n = perpendicular(v);
  c = n.dot(u);
}

// The functions below are for filtering into potential horizontal and vertical court lines
double Line::getAngle(const Line& otherLine) const
{
  double angle = abs(atan2(v.y, v.x) - atan2(otherLine.v.y, otherLine.v.x));
  angle = min(angle, CV_PI - angle);
  return angle;
}

bool Line::computeIntersectionPoint(const Line& line, cv::Point2f& intersectionPoint) const
{
  Point2f x = line.getPoint() - u;
  Point2f d1 = v;
  Point2f d2 = line.getVector();
  float cross = d1.x * d2.y - d1.y * d2.x;
  if (abs(cross) < EPS)
    return false;
  double t1 = (x.x * d2.y - x.y * d2.x) / cross;
  intersectionPoint = u + d1 * t1;
  return true;
}
