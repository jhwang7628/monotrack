#include "CourtLineCandidateDetector.h"
#include "GlobalParameters.h"
#include "TimeMeasurement.h"
#include "DebugHelpers.h"

using namespace cv;

bool CourtLineCandidateDetector::debug = false;
const std::string CourtLineCandidateDetector::windowName = "CourtLineCandidateDetector";

CourtLineCandidateDetector::Parameters::Parameters()
{
  houghThreshold = 50;
  distanceThreshold = 8;
  refinementIterations = 50;
}

CourtLineCandidateDetector::CourtLineCandidateDetector()
  : CourtLineCandidateDetector(Parameters())
{

}


CourtLineCandidateDetector::CourtLineCandidateDetector(CourtLineCandidateDetector::Parameters p)
  : parameters(p)
{

}


std::vector<Line> CourtLineCandidateDetector::run(const cv::Mat& binaryImage, const cv::Mat& rgbImage)
{
  TimeMeasurement::start("CourtLineCandidateDetector::run");

  std::vector<Line> lines;
  TimeMeasurement::start("\textractLines");
  lines = extractLines(binaryImage, rgbImage);
  TimeMeasurement::start("\textractLines");

  std::vector<std::pair<int, int>> whitePixels;
  for (int i = 0; i < binaryImage.rows; i++) {
    for (int j = 0; j < binaryImage.cols; j++) {
      if (binaryImage.at<uchar>(i, j) == GlobalParameters().fgValue) {
        whitePixels.push_back(std::make_pair(j, i));
      }
    }
  }

  for (int i = 0; i < parameters.refinementIterations; ++i)
  {
    TimeMeasurement::start("\tgetRefinedParameters");
    refineLineParameters(lines, whitePixels, rgbImage);
    TimeMeasurement::stop("\tgetRefinedParameters");

    TimeMeasurement::start("\tremoveDuplicateLines");
    removeDuplicateLines(lines, rgbImage);
    TimeMeasurement::stop("\tremoveDuplicateLines");
  }

  TimeMeasurement::stop("CourtLineCandidateDetector::run");
  return lines;
}

std::vector<Line> CourtLineCandidateDetector::extractLines(const cv::Mat& binaryImage,
  const cv::Mat& rgbImage)
{
  std::vector<cv::Vec4i> linesP;
  HoughLinesP(binaryImage, linesP, 1, CV_PI/180, parameters.houghThreshold, 50, 10);
  std::vector<Line> lines;
  for (size_t i = 0; i < linesP.size(); ++i)
  {
    cv::Point2f u(linesP[i][0], linesP[i][1]), v(linesP[i][2], linesP[i][3]);
    lines.push_back(Line::fromTwoPoints(u, v));
  }

  if (debug)
  {
    std::cout << "CourtLineCandidateDetector::extractLines line count = " << lines.size() << std::endl;
    Mat image = rgbImage.clone();
    drawLines(lines, image);
    displayImage(windowName, image);
  }

  return lines;
}


void CourtLineCandidateDetector::refineLineParameters(std::vector<Line>& lines,
  const std::vector<std::pair<int, int>>& whitePixels, const Mat& rgbImage)
{
  for (auto& line: lines)
  {
    line = getRefinedParameters(line, whitePixels, rgbImage);
  }
  if (debug)
  {
    Mat image = rgbImage.clone();
    drawLines(lines, image);
    displayImage(windowName, image);
  }
}

bool lineEqual(const Line& a, const Line& b)
{
  return a.isDuplicate(b);
}


bool CourtLineCandidateDetector::operator()(const Line& a, const Line& b)
{
  Mat tmpImage = image.clone();
  drawLine(a, tmpImage);
  drawLine(b, tmpImage);
  displayImage(windowName, tmpImage, 1);
  return a.isDuplicate(b);
}

// This is to be O(n^2) and compare all pairs of lines.
// Should be fast enough because we only expect a few thousand lines.
void CourtLineCandidateDetector::removeDuplicateLines(std::vector<Line>& lines, const cv::Mat& rgbImage)
{
  image = rgbImage.clone();
  std::vector<int> isUnique(lines.size(), -1);
  int curr_id = 0;
  for (size_t i = 0; i < lines.size(); i++) {
    if (isUnique[i] >= 0) continue;
    isUnique[i] = curr_id;
    for (size_t j = i+1; j < lines.size(); j++) {
      if (lineEqual(lines[i], lines[j])) {
        isUnique[j] = curr_id;
      }
    }
    curr_id++;
  }

  std::vector<Line> newLines(curr_id);
  std::vector<float> count(curr_id);
  for (size_t i = 0; i < lines.size(); i++) {
    newLines[isUnique[i]].u += lines[i].u;
    newLines[isUnique[i]].v += lines[i].v;
    count[isUnique[i]]++;
  }

  for (int i = 0; i < curr_id; i++) {
    // Average all lines of each group
    newLines[i].u /= count[i];
    newLines[i].v /= count[i];
  }
  lines = newLines;

  if (debug)
  {
    std::cout << "CourtLineCandidateDetector::removeDuplicateLines line count =  " << lines.size() << std::endl;
    Mat image = rgbImage.clone();
    drawLines(lines, image);
    displayImage(windowName, image);
  }
}

Line CourtLineCandidateDetector::getRefinedParameters(Line line, const std::vector<std::pair<int, int>>& whitePixels,
  const cv::Mat& rgbImage)
{
  Mat A = getClosePointsMatrix(line, whitePixels, rgbImage);
  Mat X = Mat::zeros(1, 4, CV_32F);
  fitLine(A, X, DIST_L2, 0, 0.01, 0.01);
  Point2f v(X.at<float>(0,0), X.at<float>(0,1));
  Point2f p(X.at<float>(0,2), X.at<float>(0,3));
  return Line(p, v);
}

Mat CourtLineCandidateDetector::getClosePointsMatrix(Line line, const std::vector<std::pair<int, int>>& whitePixels,
  const cv::Mat& rgbImage)
{
  Mat M = Mat::zeros(0, 2, CV_32F);

  for (const auto& p : whitePixels) {
    int x = p.first, y = p.second;
    float distance = line.getDistance(Point2f(x, y));
    if (distance < parameters.distanceThreshold)
    {
      // drawPoint(Point2f(x, y), image, Scalar(255,0,0));
      Mat point = Mat::zeros(1, 2, CV_32F);
      point.at<float>(0, 0) = x;
      point.at<float>(0, 1) = y;
      M.push_back(point);
    }
  }

  // if (true)
  // {
  //   drawLine(line, image);
  //   displayImage(windowName, image);
  // }

  return M;
}
