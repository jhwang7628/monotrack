#pragma once

#include "Line.h"
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace geometry {
	const double EPS = 1e-6;
};

float length(const cv::Point2f& v);

float distance(const cv::Point2f& p1, const cv::Point2f& p2);

float area_tri(const cv::Point2f& p1, const cv::Point2f& p2);

float area_quad(const cv::Point2f& v0, const cv::Point2f& v1, const cv::Point2f& v2, const cv::Point2f& v3);

bool inside_quad(const cv::Point2f& p, const cv::Point2f* quad);

bool seg_x_seg(cv::Point2f a, cv::Point2f b, cv::Point2f c, cv::Point2f d);

cv::Point2f perpendicular(const cv::Point2f& v);

cv::Point2f normalize(const cv::Point2f& v);

void sortLinesByDistanceToPoint(std::vector<Line>& lines, const cv::Point2f& point);

void sortLinesByLineIntersections(std::vector<Line>& lines, const Line& line);
