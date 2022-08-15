#include "geometry.h"

using namespace geometry;

float length(const cv::Point2f& v)
{
  return sqrt(v.x*v.x + v.y*v.y);
}

float cross(const cv::Point2f& p1, const cv::Point2f& p2) {
  return p1.x * p2.y - p1.y * p2.x;
}

float distance(const cv::Point2f& p1, const cv::Point2f& p2)
{
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return sqrt(dx*dx + dy*dy);
}

// Two vectors describing sides of triangle
float area_tri(const cv::Point2f& p1, const cv::Point2f& p2) {
  return abs(cross(p1, p2)) / 2.;
}

// Area of a convex quadrilateral
float area_quad(const cv::Point2f& v0, const cv::Point2f& v1, const cv::Point2f& v2, const cv::Point2f& v3) {
  return area_tri(v1-v0, v3-v0) + area_tri(v1-v2, v3-v2);
}

cv::Point2f perpendicular(const cv::Point2f& v)
{
  return cv::Point2f(-v.y, v.x);
}

cv::Point2f normalize(const cv::Point2f& v)
{
  float l = length(v);
  return cv::Point2f(v.x/l, v.y/l);
}

int sgn(const double& x) { return abs(x) < EPS ? 0 : x < 0 ? -1 : 1; }

bool compare_by_x(const cv::Point2f& a, const cv::Point2f& b) {
  return a.x < b.x-EPS || (a.x <= b.x + EPS && a.y < b.y - EPS);
}

bool seg_x_seg(cv::Point2f a, cv::Point2f b, cv::Point2f c, cv::Point2f d) {
  if(distance(a, b) < EPS || distance(c, d) < EPS) return 0; // exclude endpoints
  double sa=length(b-a), sc=length(d-c); sa=sa>EPS?1/sa:0; sc=sc>EPS?1/sc:0;
  int r1 = sgn(cross(b-a, c-a) * sa), r2 = sgn(cross(b-a, d-a) * sa);
  int r3 = sgn(cross(d-c, a-c) * sc), r4 = sgn(cross(d-c, b-c) * sc);
  if(!r1 && !r2 && !r3) { // collinear
    if(compare_by_x(b, a)) swap(a, b);
    if(compare_by_x(d, c)) swap(c, d);
    return compare_by_x(a, d) && compare_by_x(c, b); // exclude endpoints
    return !compare_by_x(d, a) && !compare_by_x(b, c);
  } return r1*r2 < 0 && r3*r4 < 0; 
}

class LineComparator
{
public:
  LineComparator(cv::Point2f point) : p(point) { }

  bool operator()(const Line& lineA, const Line& lineB)
  {
    return lineA.getDistance(p) < lineB.getDistance(p);
  }

private:
  cv::Point2f p;
};

class LineIntersectionComparator
{
public:
  LineIntersectionComparator(Line line) : l(line) { }

  bool operator()(const Line& lineA, const Line& lineB)
  {
    cv::Point2f p, q;
    if (l.getAngle(lineA) > Line::EPS) {
      l.computeIntersectionPoint(lineA, p);
    } else {
      p = lineA.getPointOnLineClosestTo(l.u);
    }

    if (l.getAngle(lineB) > Line::EPS) {
      l.computeIntersectionPoint(lineB, q);
    } else {
      q = lineB.getPointOnLineClosestTo(l.u);
    }

    return distance(p, l.u) < distance(q, l.u);
  }

private:
  Line l;
};

void sortLinesByDistanceToPoint(std::vector<Line>& lines, const cv::Point2f& point)
{
  std::sort(lines.begin(), lines.end(), LineComparator(point));
}

void sortLinesByLineIntersections(std::vector<Line>& lines, const Line& line)
{
  std::sort(lines.begin(), lines.end(), LineIntersectionComparator(line));
}