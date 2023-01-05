//
// Created by Chlebus, Grzegorz on 28.08.17.
// Copyright (c) Chlebus, Grzegorz. All rights reserved.
//
#pragma once

#include "Line.h"
#include <opencv2/opencv.hpp>
#include "BadmintonCourtModel.h"

class BadmintonCourtFitter
{
public:
  struct Parameters
  {
    Parameters();
  };

  BadmintonCourtFitter();

  BadmintonCourtFitter(Parameters p);

  BadmintonCourtModel run(const std::vector<Line>& lines, const cv::Mat& binaryImage, const cv::Mat& rgbImage);

  static bool debug;
  static const std::string windowName;

private:
  void getHorizontalAndVerticalLines(const std::vector<Line>& lines, std::vector<Line>& hLines,
    std::vector<Line>& vLines, const cv::Mat& rgbImage, int mode=1);

  void sortHorizontalLines(std::vector<Line>& hLines, const cv::Mat& rgbImage);

  void sortVerticalLines(std::vector<Line>& vLines, const cv::Mat& rgbImage);

  void findBestModelFit(const std::vector<Line>& lines, const cv::Mat& binaryImage, const cv::Mat& rgbImage, int mode);

  Parameters parameters;
  std::vector<LinePair> hLinePairs;
  std::vector<LinePair> vLinePairs;
  BadmintonCourtModel bestModel;
  float bestScore;
};