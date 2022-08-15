#include "BadmintonCourtFitter.h"
#include "GlobalParameters.h"
#include "TimeMeasurement.h"
#include "DebugHelpers.h"
#include "geometry.h"


using namespace cv;

bool BadmintonCourtFitter::debug = false;
const std::string BadmintonCourtFitter::windowName = "BadmintonCourtFitter";

BadmintonCourtFitter::Parameters::Parameters()
{

}

BadmintonCourtFitter::BadmintonCourtFitter()
  : BadmintonCourtFitter(Parameters())
{

}


BadmintonCourtFitter::BadmintonCourtFitter(BadmintonCourtFitter::Parameters p)
  : parameters(p)
{

}

BadmintonCourtModel BadmintonCourtFitter::run(const std::vector<Line>& lines, const Mat& binaryImage,
  const Mat& rgbImage)
{
  TimeMeasurement::start("BadmintonCourtFitter::run");

  bestScore = GlobalParameters().initialFitScore;

  TimeMeasurement::start("\tfindBestModelFit (optimization mode)");
  findBestModelFit(lines, binaryImage, rgbImage, 1);
  std::cerr << "Current best model score: " << bestScore << std::endl;
  TimeMeasurement::stop("\tfindBestModelFit (optimization mode)");

  const int minThresh = 2 * 1600, goodThresh = 2 * 2100;
  if (bestScore < minThresh) {
    std::cout << "Fit scores in default-mode too low. Trying more time-intensive computations..." << std::endl;
    TimeMeasurement::start("\tfindBestModelFit (random mode)");
    const int trials = 24;
    for (int t = 0; t < trials; t++) {
      findBestModelFit(lines, binaryImage, rgbImage, 0);
      if (t % 5 == 0) {
        std::cerr << "Iteration " << t << ", current best model score: " 
                  << bestScore << std::endl;
      }
      if (bestScore >= goodThresh) {
        break;
      }
    }
    TimeMeasurement::stop("\tfindBestModelFit (random mode)");
  }

  TimeMeasurement::stop("BadmintonCourtFitter::run");

  return bestModel;
}


void BadmintonCourtFitter::getHorizontalAndVerticalLines(const std::vector<Line>& lines,
  std::vector<Line>& hLines, std::vector<Line>& vLines, const cv::Mat& rgbImage, int mode)
{
  std::vector<int> bestColour;
  if (mode == 0) {
    // Random assignment mode
    bestColour.resize(lines.size());
    for (size_t i = 0; i < lines.size(); i++) {
      bestColour[i] = rand() % 2;
    }
  } else {
    // Optimization-based mode
    // Create a graph between lines:
    //  - edge between lines if they are separated by at least 20 deg.
    //  - two colour this graph to obtain vertical and horizontal lines
    std::vector<std::vector<double>> weights;
    for (size_t i = 0; i < lines.size(); i++) {
      weights.push_back(std::vector<double>(lines.size()));
      for (size_t j = 0; j < lines.size(); j++) {
        double angle = lines[i].getAngle(lines[j]);
        double weight = 1./pow(1 - 2 * angle / CV_PI, 2) - 1;
        weights[i][j] = weight;
      }
    }

    // Arbitrarily go through vertices, assign to other side if has more white neighbours
    // assigned than black
    double bestCost = 1e99;

    const int trials = 42;
    for (int t = 0; t < trials; t++) {
      std::vector<int> colour(lines.size());
      for (size_t i = 0; i < lines.size(); i++) {
        colour[i] = rand() % 2;
      }

      double conflicts = 1e99;
      while (true) {
        // Strategy: two-opt and then one-opt search.
        bool swapped = true;
        int iters = 10;
        while (swapped && --iters) {
          swapped = false;
          for (size_t i = 0; i < lines.size(); i++) {
            for (size_t j = 0; j < lines.size(); j++) {
              if (colour[i] ^ colour[j]) {
                // How much do we gain from swapping?
                double conflict_i[2] = {0, 0};
                double conflict_j[2] = {0, 0};
                for (size_t k = 0; k < lines.size(); k++) {
                  if (k == j || k == i) continue;
                  conflict_i[colour[i] ^ colour[k]] += weights[i][k];
                  conflict_j[colour[j] ^ colour[k]] += weights[j][k];
                }

                // If swap, then the overall conflict changes by:
                // -c_i[0]-c_j[0]+c_i[1]+c_j[1]
                double delta = conflict_i[1] + conflict_j[1] - conflict_i[0] - conflict_j[0];
                if (delta < 0) {
                  colour[i] ^= 1;
                  colour[j] ^= 1;
                  swapped = true;
                }
              }
            }
          }
        }

        for (size_t i = 0; i < lines.size(); i++) {
          double n[2] = {0, 0};
          for (size_t j = 0; j < lines.size(); j++) {
            n[colour[j] ^ colour[i]] += weights[i][j];
          }

          if (n[0] > n[1]) {
            colour[i] ^= 1;
          }
        }

        double new_conflicts = 0;
        for (size_t i = 0; i < lines.size(); i++) {
          for (size_t j = 0; j < lines.size(); j++) {
            if (colour[i] == colour[j]) {
              new_conflicts += weights[i][j];
            }
          }
        }

        if (new_conflicts < conflicts) {
          conflicts = new_conflicts;
        } else {
          break;
        }
      }

      if (conflicts < bestCost) {
        bestCost = conflicts;
        bestColour = colour;
      }
    }

    std::cerr << "Minimum conflict achieved: " << bestCost << std::endl;
  }

  for (size_t i = 0; i < lines.size(); i++) {
    if (bestColour[i] == 0) {
      vLines.push_back(lines[i]);
    } else {
      hLines.push_back(lines[i]);
    }
  }

  if (debug)
  {
    std::cout << "Horizontal lines = " << hLines.size() << std::endl;
    std::cout << "Vertical lines = " << vLines.size() << std::endl;
    Mat image = rgbImage.clone();
    drawLines(hLines, image, Scalar(255, 0, 0));
    drawLines(vLines, image, Scalar(0, 255, 0));
    displayImage(windowName, image);
  }
}


void BadmintonCourtFitter::sortHorizontalLines(std::vector<Line>& hLines, const cv::Mat& rgbImage)
{
  auto line = Line::fromTwoPoints(Point2f(rgbImage.cols / 2, rgbImage.rows), Point2f(0, 1));
  sortLinesByLineIntersections(hLines, line);

  if (false)
  {
    for (auto& line: hLines)
    {
      Mat image = rgbImage.clone();
      drawLine(line, image, Scalar(255, 0, 0));
      displayImage(windowName, image);
    }
  }
}

void BadmintonCourtFitter::sortVerticalLines(std::vector<Line>& vLines, const cv::Mat& rgbImage)
{
  // TODO: We should be computing the hulls formed by the region below the lines
  // and picking a point in the hull, but this should be good enough.
  auto line = Line::fromTwoPoints(Point2f(0, rgbImage.rows / 2), Point2f(1, 0));
  sortLinesByLineIntersections(vLines, line);

  if (false)
  {
    for (auto& line: vLines)
    {
      Mat image = rgbImage.clone();
      drawLine(line, image, Scalar(0, 255, 0));
      displayImage(windowName, image);
    }
  }
}


void BadmintonCourtFitter::findBestModelFit(const std::vector<Line>& lines, const cv::Mat& binaryImage, const cv::Mat& rgbImage, int mode)
{
  std::vector<Line> hLines, vLines;
  getHorizontalAndVerticalLines(lines, hLines, vLines, rgbImage, mode);

  for (int flip = 0; flip < 2; flip++) {
    if (flip) {
      hLines.swap(vLines);
    }

    sortHorizontalLines(hLines, rgbImage);
    sortVerticalLines(vLines, rgbImage);

    hLinePairs = BadmintonCourtModel::getPossibleLinePairs(hLines);
    vLinePairs = BadmintonCourtModel::getPossibleLinePairs(vLines);

    if (debug)
    {
      std::cout << "Horizontal line pairs = " << hLinePairs.size() << std::endl;
      std::cout << "Vertical line pairs = " << vLinePairs.size() << std::endl;
    }

    if (hLinePairs.empty() || vLinePairs.empty())
    {
      throw std::runtime_error("Not enough line candidates were found.");
    }

    for (auto& hLinePair: hLinePairs)
    {
      for (auto& vLinePair: vLinePairs)
      {
        BadmintonCourtModel model;
        float score = model.fit(hLinePair, vLinePair, binaryImage, rgbImage);
        float netScore = 0;
        if (score > GlobalParameters().initialFitScore) {
          netScore = model.fitNet(lines, binaryImage, rgbImage);
        }
        
        // TODO: Figure out how to score the white pixels of the net
        if (score + 1e-2 * netScore > bestScore)
        {
          bestScore = score + 1e-2 * netScore;
          bestModel = model;
          std::cerr << "Score breakdown: " << score << " " << netScore << std::endl;
          if (debug) {
            Mat image = rgbImage.clone();
            drawLine(hLinePair.first, image, Scalar(255, 0, 0));
            drawLine(hLinePair.second, image, Scalar(255, 0, 0));
            drawLine(vLinePair.first, image, Scalar(255, 0, 0));
            drawLine(vLinePair.second, image, Scalar(255, 0, 0));
            displayImage(windowName, image);

            bestModel.drawModel(image);
            displayImage(windowName, image);
          }
        }
      }
    }
  }

  if (debug)
  {
    std::cout << "Best model score = " << bestScore << std::endl;
    Mat image = rgbImage.clone();
    bestModel.drawModel(image);
    displayImage(windowName, image);
  }
}
