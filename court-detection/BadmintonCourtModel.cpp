#include "BadmintonCourtModel.h"
#include "GlobalParameters.h"
#include "DebugHelpers.h"
#include "geometry.h"
#include "TimeMeasurement.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

using namespace cv;

BadmintonCourtModel::BadmintonCourtModel()
{
  Point2f hVector(1, 0);
  const Line upperBaseLine = Line(Point2f(0, 0), hVector);
  const Line upperDoublesLine = Line(Point2f(0, 0.76), hVector);
  const Line upperServiceLine = Line(Point2f(0, 4.72), hVector);
  const Line netLine = Line(Point2f(0, 6.705), hVector);
  const Line lowerServiceLine = Line(Point2f(0, 8.685), hVector);
  const Line lowerDoublesLine = Line(Point2f(0, 12.65), hVector);
  const Line lowerBaseLine = Line(Point2f(0, 13.41), hVector);
  // hLines = {
  //   upperBaseLine, upperDoublesLine, upperServiceLine, netLine, lowerServiceLine, lowerDoublesLine, lowerBaseLine
  // };
  hLines = {
    lowerBaseLine, lowerDoublesLine, lowerServiceLine, netLine, upperServiceLine, upperDoublesLine, upperBaseLine
  };

  Point2f vVector(0, 1);
  const Line leftSideLine = Line(Point2f(0, 0), vVector);
  const Line leftSinglesLine = Line(Point2f(0.46, 0), vVector);
  const Line centreServiceLine = Line(Point2f(3.05, 0), vVector);
  const Line rightSinglesLine = Line(Point2f(5.64, 0), vVector);
  const Line rightSideLine = Line(Point2f(6.1, 0), vVector);
  vLines = {
    leftSideLine, leftSinglesLine, centreServiceLine, rightSinglesLine, rightSideLine
  };

  // Uncomment for more complete line checking
  hLinePairs = getPossibleLinePairs(hLines);
  vLinePairs = getPossibleLinePairs(vLines);
  // hLinePairs.push_back(std::make_pair(hLines[2], hLines[4]));
  // vLinePairs.push_back(std::make_pair(vLines[1], vLines[3]));

  // // Otherwise only look for the following boxes
  // // Doubles boxes
  // hLinePairs.push_back(std::make_pair(hLines[1], hLines[2]));
  // hLinePairs.push_back(std::make_pair(hLines[4], hLines[5]));
  // hLinePairs.push_back(std::make_pair(hLines[1], hLines[5]));
  // vLinePairs.push_back(std::make_pair(vLines[1], vLines[2]));
  // vLinePairs.push_back(std::make_pair(vLines[2], vLines[3]));

  // // Entire court
  // hLinePairs.push_back(std::make_pair(hLines[0], hLines[6]));
  // vLinePairs.push_back(std::make_pair(vLines[0], vLines[4]));

  // // Singles court
  // vLinePairs.push_back(std::make_pair(vLines[1], vLines[3]));

  // // Combinations of side lines
  // hLinePairs.push_back(std::make_pair(hLines[0], hLines[5]));
  // hLinePairs.push_back(std::make_pair(hLines[1], hLines[6]));
  // vLinePairs.push_back(std::make_pair(vLines[1], vLines[4]));
  // vLinePairs.push_back(std::make_pair(vLines[0], vLines[3]));

  Point2f point;
  if (upperBaseLine.computeIntersectionPoint(leftSideLine, point))
  {
    courtPoints.push_back(point); // P1
  }
  if (lowerBaseLine.computeIntersectionPoint(leftSideLine, point))
  {
    courtPoints.push_back(point); // P2
  }
  if (lowerBaseLine.computeIntersectionPoint(rightSideLine, point))
  {
    courtPoints.push_back(point); // P3
  }
  if (upperBaseLine.computeIntersectionPoint(rightSideLine, point))
  {
    courtPoints.push_back(point);  // P4
  }
  if (upperBaseLine.computeIntersectionPoint(leftSinglesLine, point))
  {
    courtPoints.push_back(point);  // P5
  }
  if (lowerBaseLine.computeIntersectionPoint(leftSinglesLine, point))
  {
    courtPoints.push_back(point);  // P6
  }
  if (lowerBaseLine.computeIntersectionPoint(rightSinglesLine, point))
  {
    courtPoints.push_back(point);  // P7
  }
  if (upperBaseLine.computeIntersectionPoint(rightSinglesLine, point))
  {
    courtPoints.push_back(point);  // P8
  }
  if (leftSideLine.computeIntersectionPoint(upperServiceLine, point))
  {
    courtPoints.push_back(point);  // P9
  }
  if (rightSideLine.computeIntersectionPoint(upperServiceLine, point))
  {
    courtPoints.push_back(point);  // P10
  }
  if (leftSideLine.computeIntersectionPoint(lowerServiceLine, point))
  {
    courtPoints.push_back(point);  // P11
  }
  if (rightSideLine.computeIntersectionPoint(lowerServiceLine, point))
  {
    courtPoints.push_back(point);  // P12
  }
  if (upperServiceLine.computeIntersectionPoint(centreServiceLine, point))
  {
    courtPoints.push_back(point);  // P13
  }
  if (lowerServiceLine.computeIntersectionPoint(centreServiceLine, point))
  {
    courtPoints.push_back(point);  // P14
  }
  if (leftSideLine.computeIntersectionPoint(netLine, point))
  {
    courtPoints.push_back(point);  // P15
  }
  if (rightSideLine.computeIntersectionPoint(netLine, point))
  {
    courtPoints.push_back(point);  // P16
  }
  if (leftSideLine.computeIntersectionPoint(upperDoublesLine, point))
  {
    courtPoints.push_back(point);  // P17
  }
  if (rightSideLine.computeIntersectionPoint(upperDoublesLine, point))
  {
    courtPoints.push_back(point);  // P18
  }
  if (leftSideLine.computeIntersectionPoint(lowerDoublesLine, point))
  {
    courtPoints.push_back(point);  // P19
  }
  if (rightSideLine.computeIntersectionPoint(lowerDoublesLine, point))
  {
    courtPoints.push_back(point);  // P20
  }
  if (upperBaseLine.computeIntersectionPoint(centreServiceLine, point))
  {
    courtPoints.push_back(point);  // P21
  }
  if (lowerBaseLine.computeIntersectionPoint(centreServiceLine, point))
  {
    courtPoints.push_back(point);  // P22
  }

  assert(courtPoints.size() == 22);
}

BadmintonCourtModel::BadmintonCourtModel(const BadmintonCourtModel& o)
  : transformationMatrix(o.transformationMatrix)
{
  courtPoints = o.courtPoints;
  hLinePairs = o.hLinePairs;
  vLinePairs = o.vLinePairs;
  hLines = o.hLines;
  vLines = o.vLines;
  netPoints = o.netPoints;
}

BadmintonCourtModel& BadmintonCourtModel::operator=(const BadmintonCourtModel& o)
{
  assert(o.netPoints.size() == 2);
  transformationMatrix = o.transformationMatrix;
  netPoints = o.netPoints;
  return *this;
}

float BadmintonCourtModel::fit(const LinePair& hLinePair, const LinePair& vLinePair,
  const cv::Mat& binaryImage, const cv::Mat& rgbImage)
{
  float bestScore = GlobalParameters().initialFitScore;
  std::vector<Point2f> points = getIntersectionPoints(hLinePair, vLinePair);

  // If not enough intersections
  if (points.size() < 4) return bestScore;


  // If there are crossings, return immediately
  if (seg_x_seg(points[0], points[1], points[2], points[3]) ||
      seg_x_seg(points[1], points[2], points[3], points[0])) {
    return bestScore;
  }

  // If the shape is concave, then theres no point either
  if (!isContourConvex(points)) {
    return bestScore;
  }

  // Mat image = rgbImage.clone();
  // // drawModel(transformedModelPoints, image);
  // drawLine(points[0], points[1], image, 0);
  // drawLine(points[1], points[2], image, 0);
  // drawLine(points[2], points[3], image, 0);
  // drawLine(points[3], points[0], image, 0);
  // displayImage("BadmintonCourtModel", image, 0);
  
  for (auto& modelHLinePair: hLinePairs)
  {
    for (auto& modelVLinePair: vLinePairs)
    {
      std::vector<Point2f> modelPoints = getIntersectionPoints(modelHLinePair, modelVLinePair);
      Mat matrix = getPerspectiveTransform(modelPoints, points);
      std::vector<Point2f> transformedModelPoints(courtPoints.size());

      perspectiveTransform(courtPoints, transformedModelPoints, matrix);
      float score = evaluateModel(transformedModelPoints, binaryImage);
      if (score > bestScore)
      {
        bestScore = score;
        transformationMatrix = matrix;
      }
    }
  }
  return bestScore;
}

float BadmintonCourtModel::fitNet(const std::vector<Line>& lines, const cv::Mat& binaryImage, 
  const cv::Mat& rgbImage) {
  // Filter the lines for candidates
  std::vector<Line> filteredLines;
  
  std::vector<Point2f> transformedModelPoints(courtPoints.size());
  perspectiveTransform(courtPoints, transformedModelPoints, transformationMatrix);
  Line middleLine = Line::fromTwoPoints(transformedModelPoints[14], transformedModelPoints[15]);
  
  int linePairs[7][2] = {
    //{0, 3}, {1, 2}, 
    {8, 9}, {10, 11}, {16, 17}, {18, 19}, {14, 15}
  };
  std::vector<Line> excludedLines;
  for (int i = 0; i < 7; i++) {
    auto line = Line::fromTwoPoints(transformedModelPoints[linePairs[i][0]], transformedModelPoints[linePairs[i][1]]);
    excludedLines.push_back(line);
  }

  int courtPairs[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {6, 7}, {8, 9}, {10, 11},
    {12, 20}, {13, 21}, {16, 17}, {18, 19}
  };

  // Remove pixels from lines that are part of the court
  cv::Mat filteredBinaryImage = binaryImage.clone();
  for (int i = 0; i < 12; i++) {
    removeLineSegment(
      transformedModelPoints[courtPairs[i][0]], 
      transformedModelPoints[courtPairs[i][1]],
      filteredBinaryImage
    );
  }


  for (const auto& line : lines) {
    if (middleLine.isParallel(line, 0.2)) {
      bool excluded = false;
      for (const auto& exclLine : excludedLines) {
        if (exclLine.isDuplicate(line)) {
          excluded = true;
          break;
        }
      }

      // Check if this line is sufficiently above the middle line
      double xmid = (transformedModelPoints[14].x + transformedModelPoints[15].x) / 2.;
      // Bound net between 5% and and 40% of the screen 
      double pixThreshLow = 0.05 * binaryImage.rows, pixThreshHigh = 0.4 * binaryImage.rows;
      if (!excluded && line.evaluateByX(xmid) < middleLine.evaluateByX(xmid) - pixThreshLow &&
          line.evaluateByX(xmid) > middleLine.evaluateByX(xmid) - pixThreshHigh) {
        filteredLines.push_back(line);
      }
    }
  }

  if (filteredLines.size() == 0) {
    return GlobalParameters().initialFitScore;
  }

  Line bestLine;
  float bestScore = GlobalParameters().initialFitScore;

  for (const auto& line : filteredLines) {
    auto leftPost = line.getPointOnLineClosestTo(transformedModelPoints[14]);
    auto rightPost = line.getPointOnLineClosestTo(transformedModelPoints[15]);
    float length = computeRasterizedSegmentLength(leftPost, rightPost, filteredBinaryImage);

    // If less than half the net is in the frame, this is probably wrong
    if (length < 0.5 * distance(leftPost, rightPost)) {
      continue;
    }

    // It should not be too short
    if (distance(leftPost, transformedModelPoints[14]) > length * 0.33 ||
        distance(rightPost, transformedModelPoints[15]) > length * 0.33) {
      continue;
    }

    // It should not be too short either
    if (distance(leftPost, transformedModelPoints[14]) < length * 0.1 ||
        distance(rightPost, transformedModelPoints[15]) < length * 0.1) {
      continue;
    }

    double w_net = 4;
    float score = computeScoreForLineSegment(leftPost, rightPost, binaryImage, w_net);
    if (score > bestScore) {
      bestScore = score;
      bestLine = line;
      netPoints = std::vector<cv::Point2f>({leftPost, rightPost});
    }
  }

  return bestScore;
}

std::vector<cv::Point2f> BadmintonCourtModel::getIntersectionPoints(const LinePair& hLinePair,
  const LinePair& vLinePair)
{
  std::vector<Point2f> v;
  Point2f point;

  if (hLinePair.first.computeIntersectionPoint(vLinePair.first, point))
  {
    v.push_back(point);
  }
  if (hLinePair.first.computeIntersectionPoint(vLinePair.second, point))
  {
    v.push_back(point);
  }
  if (hLinePair.second.computeIntersectionPoint(vLinePair.second, point))
  {
    v.push_back(point);
  }
  if (hLinePair.second.computeIntersectionPoint(vLinePair.first, point))
  {
    v.push_back(point);
  }

  // assert(v.size() == 4);

  return v;
}

std::vector<LinePair> BadmintonCourtModel::getPossibleLinePairs(std::vector<Line>& lines)
{
  std::vector<LinePair> linePairs;
  for (size_t first = 0; first < lines.size(); ++first)
//  for (size_t first = 0; first < 1; ++first)
  {
    for (size_t second = first + 1; second < lines.size(); ++second)
    {
      linePairs.push_back(std::make_pair(lines[first], lines[second]));
    }
  }
  return linePairs;
}


void BadmintonCourtModel::drawModel(cv::Mat& image, Scalar color)
{
  std::vector<Point2f> transformedModelPoints(courtPoints.size());
  perspectiveTransform(courtPoints, transformedModelPoints, transformationMatrix);
  drawModel(transformedModelPoints, image, color);
}

void BadmintonCourtModel::drawModel(std::vector<Point2f>& courtPoints, Mat& image, Scalar color)
{
  // Outside box
  drawLine(courtPoints[0], courtPoints[1], image, color);
  drawLine(courtPoints[1], courtPoints[2], image, color);
  drawLine(courtPoints[2], courtPoints[3], image, color);
  drawLine(courtPoints[3], courtPoints[0], image, color);

  // Left and right singles line
  drawLine(courtPoints[4], courtPoints[5], image, color);
  drawLine(courtPoints[6], courtPoints[7], image, color);

  // Front service lines
  drawLine(courtPoints[8], courtPoints[9], image, color);
  drawLine(courtPoints[10], courtPoints[11], image, color);

  // Net line
  // drawLine(courtPoints[14], courtPoints[15], image, color);

  // Middle line
  drawLine(courtPoints[12], courtPoints[20], image, color);
  drawLine(courtPoints[13], courtPoints[21], image, color);

  // Back doubles service lines
  drawLine(courtPoints[16], courtPoints[17], image, color);
  drawLine(courtPoints[18], courtPoints[19], image, color);

  // The net
  if (netPoints.size()) {
    drawLine(netPoints[0], netPoints[1], image, Scalar(0, 255, 0));
    drawLine(courtPoints[14], netPoints[0], image, Scalar(0, 255, 0));
    drawLine(courtPoints[15], netPoints[1], image, Scalar(0, 255, 0));
    // drawLine(courtPoints[14], courtPoints[15], image, Scalar(0, 255, 0));
  }
}


float BadmintonCourtModel::evaluateModel(const std::vector<cv::Point2f>& courtPoints, const cv::Mat& binaryImage)
{
  float score = 0;

  // Heuristics to see whether the model makes sense
  float d1 = distance(courtPoints[0], courtPoints[1]);
  float d2 = distance(courtPoints[1], courtPoints[2]);
  float d3 = distance(courtPoints[2], courtPoints[3]);
  float d4 = distance(courtPoints[3], courtPoints[0]);
  float t = 168;
  if (d1 < t || d2 < t || d3 < t || d4 < t)
  {
    return GlobalParameters().initialFitScore;
  }

  // If there are crossings, skip
  if (seg_x_seg(courtPoints[0], courtPoints[1], courtPoints[2], courtPoints[3]) ||
      seg_x_seg(courtPoints[1], courtPoints[2], courtPoints[3], courtPoints[0])) {
    return GlobalParameters().initialFitScore;
  }

  // If more than two points of the court are off screen, then skip
  // If a point is really off (hundreds of pixels off screen), then skip
  int off_screen = 0;
  for (int i = 0; i < 4; i++) {
    double x0 = courtPoints[i].x, y0 = courtPoints[i].y;
    if (x0 < 0 || x0 >= binaryImage.cols ||
        y0 < 0 || y0 >= binaryImage.rows) {
      off_screen++;
    }
  }

  if (off_screen >= 3) {
    return GlobalParameters().initialFitScore;
  }

  int really_off = 0;
  const int outThresh = 512;
  for (size_t i = 0; i < courtPoints.size(); i++) {
    if (courtPoints[i].x < -outThresh || courtPoints[i].y < -outThresh ||
        courtPoints[i].x >= binaryImage.cols + outThresh || courtPoints[i].y >= binaryImage.rows + outThresh) {
      really_off = 1;
    }
  }
  if (really_off) {
    return GlobalParameters().initialFitScore;
  }

  // Estimate the area on screen by clipping the coordinates to
  // the nearest screen point
  std::vector<cv::Point2f> screenPoints;
  for (int i = 0; i < 4; i++) {
    if (courtPoints[i].x < 0 || courtPoints[i].x >= binaryImage.cols ||
        courtPoints[i].y < 0 || courtPoints[i].y >= binaryImage.rows) {
      cv::Point2f p = courtPoints[i];
      p.x = std::min((float) binaryImage.cols, std::max(0.f, p.x));
      p.y = std::min((float) binaryImage.rows, std::max(0.f, p.y));
      screenPoints.push_back(p);
    } else {
      screenPoints.push_back(courtPoints[i]);
    }
  }
  float area = area_quad(screenPoints[0], screenPoints[1], screenPoints[2], screenPoints[3]);

  if (area < 0.1 * binaryImage.rows * binaryImage.cols) {
    // Immediately return if court dimensions too small
    return GlobalParameters().initialFitScore;
  }

  // If any inside court points are outside, return bad score
  std::vector<Point2f> quad({courtPoints[0], courtPoints[1], courtPoints[2], courtPoints[3]});
  for (size_t i = 4; i < courtPoints.size(); i++) {
    if (pointPolygonTest(quad, courtPoints[i], true) < -Line::PIXEL_EPS) {
      return GlobalParameters().initialFitScore;
    }
  }

  // Outside box
  double w_out = 1.25, w_singles = 2, w_front = 1, w_mid = 1, w_back = 1;
  score += computeScoreForLineSegment(courtPoints[0], courtPoints[1], binaryImage, w_out);
  score += computeScoreForLineSegment(courtPoints[1], courtPoints[2], binaryImage, w_out);
  score += computeScoreForLineSegment(courtPoints[2], courtPoints[3], binaryImage, w_out);
  score += computeScoreForLineSegment(courtPoints[3], courtPoints[0], binaryImage, w_out);

  score += computeScoreForLineSegment(courtPoints[4], courtPoints[5], binaryImage, w_singles);
  score += computeScoreForLineSegment(courtPoints[6], courtPoints[7], binaryImage, w_singles);

  score += computeScoreForLineSegment(courtPoints[8], courtPoints[9], binaryImage, w_front);
  score += computeScoreForLineSegment(courtPoints[10], courtPoints[11], binaryImage, w_front);

  score += computeScoreForLineSegment(courtPoints[12], courtPoints[20], binaryImage, w_mid);
  score += computeScoreForLineSegment(courtPoints[13], courtPoints[21], binaryImage, w_mid);

  score += computeScoreForLineSegment(courtPoints[16], courtPoints[17], binaryImage, w_back);
  score += computeScoreForLineSegment(courtPoints[18], courtPoints[19], binaryImage, w_back);

//  std::cout << "Score = " << score << std::endl;

  return score;
}

bool BadmintonCourtModel::pruneStartEnd(cv::Point2f& start, cv::Point2f& end, const cv::Mat& binaryImage) {
  // Reset start and end to intersections
  if (start.x > end.x) {
    swap(start, end);
  }

  if (abs(start.x - end.x) < Line::EPS && start.y > end.y) {
    swap(start, end);
  }

  auto dir = end - start;
  if (start.x < 0 && abs(dir.x) > Line::EPS) {
    // Move start to the boundary
    double t = -start.x / dir.x;
    start = start + t * dir;
  }

  if (start.y < 0 && abs(dir.y) > Line::EPS) {
    double t = -start.y / dir.y;
    if (t < 0) return false;
    start = start + t * dir;
  }

  if (start.y >= binaryImage.rows - 1 && abs(dir.y) > Line::EPS) {
    double t = (binaryImage.rows - 1 - start.y) / dir.y;
    if (t < 0) return false;
    start = start + t * dir;
  }

  if (end.x >= binaryImage.cols - 1 && abs(dir.x) > Line::EPS) {
    double t = (binaryImage.cols - 1 - end.x) / dir.x;
    end = end + t * dir;
  }

  if (end.y >= binaryImage.rows - 1 && abs(dir.y) > Line::EPS) {
    double t = (binaryImage.rows - 1 - end.y) / dir.y;
    if (t > 0) return false;
    end = end + t * dir;
  }

  if (end.y < 0 && abs(dir.y) > Line::EPS) {
    double t = -end.y / dir.y;
    if (t > 0) return false;
    end = end + t * dir;
  }

  if (start.x > end.x) return false;
  int length = int(distance(start, end));
  if (length == 0) return false;

  return true;
}

float BadmintonCourtModel::computeScoreForLineSegment(cv::Point2f start, cv::Point2f end,
  const cv::Mat& binaryImage, double weight)
{
  float score = 0;
  float fgScore = 1;
  float bgScore = -0.5;

  if (!pruneStartEnd(start, end, binaryImage))
    return 0;

  int length = int(distance(start, end));
  Point2f vec = normalize(end-start);

  for (int i = 0; i < length; i++)
  {
    Point2f p = start + i*vec;
    int x = round(p.x);
    int y = round(p.y);
    uchar imageValue = binaryImage.at<uchar>(y,x);
    if (imageValue == GlobalParameters().fgValue)
    {
      score += fgScore;
    }
    else
    {
      score += bgScore / weight;
    }

    /*
    // TODO: Add some width to the scoring function smartly
    // to make score function take in a line with thickness
    int dx[] = {0, 1, 0, -1, 0};
    int dy[] = {1, 0, -1, 0, 0};
    for (int j = 0; j < 5; j++) {
      if (!isInsideTheImage(y+dy[j],x+dx[j], binaryImage))
        continue;
      uchar imageValue = binaryImage.at<uchar>(y+dy[j],x+dx[j]);
      if (imageValue == GlobalParameters().fgValue)
      {
        score += fgScore;
      }
      else
      {
        score += bgScore / weight;
      }
    }
    */
  }
  return weight * score;
}

void BadmintonCourtModel::removeLineSegment(cv::Point2f start, cv::Point2f end,
  cv::Mat& binaryImage)
{
  if (!pruneStartEnd(start, end, binaryImage))
    return;

  int length = int(distance(start, end));
  Point2f vec = normalize(end-start);

  for (int i = 0; i < length; i++)
  {
    Point2f p = start + i*vec;
    int x = round(p.x);
    int y = round(p.y);
    binaryImage.at<uchar>(y,x) = 0;
  }
}

float BadmintonCourtModel::computeRasterizedSegmentLength(cv::Point2f start, cv::Point2f end,
  const cv::Mat& binaryImage)
{
  if (!pruneStartEnd(start, end, binaryImage))
    return 0;

  return distance(start, end);
}


bool BadmintonCourtModel::isInsideTheImage(float x, float y, const cv::Mat& image)
{
  return (x >= 0 && x < image.cols) && (y >= 0 && y < image.rows);
}

void BadmintonCourtModel::writeToFile(const std::string& filename)
{
  std::vector<Point2f> transformedModelPoints(courtPoints.size());
  perspectiveTransform(courtPoints, transformedModelPoints, transformationMatrix);

  std::ofstream outFile(filename);
  if (!outFile.is_open())
  {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  // Write the four corners plus the two net poles
  for (int i = 0; i < 4; i++) {
    auto point = transformedModelPoints[i];
    outFile << point.x << ";" << point.y << std::endl;
  }
  for (int i = 0; i < 2; i++) {
    auto point = netPoints[i];
    outFile << point.x << ";" << point.y << std::endl;
  }
}
