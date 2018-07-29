#include <opencv2/opencv.hpp>
#include <iostream>

const std::string ROOT_SAMPLE = "/home/ros/Projects/Calculate_map_freespace";
const int THRESHOLD = 210;
const int UNKNOWN = 200;
const float RESOLUTION = 0.02;
const int CANNY_THRESHOLD_UP = 100;
const int CANNY_THRESHOLD_DOWN = 200;
float SCALE = 0.3;
float OBSTACLE_FILTER_THRESH = 10;
float FREESPACE_FILTER_THRESH = 5000;

using namespace std;
using namespace cv;

cv::RNG rng(12345);

void remove_small_components(const cv::Mat& input, cv::Mat& output,
                             int area_thresh)
{
  cv::Mat labels, status, centroids;
  int nccomps =
      cv::connectedComponentsWithStats(input, labels, status, centroids);

  vector<uchar> colors(nccomps);
  colors[0] = 0; // background pixels remain black.
  for (int i = 1; i < nccomps; i++)
  {
    colors[i] = 255;
    //去除面积小于area_thresh的连通域
    if (status.at<int>(i, cv::CC_STAT_AREA) < area_thresh)
      colors[i] = 0; // small regions are painted with black too.
  }
  //按照label值，对不同的连通域进行着色
  output = cv::Mat::zeros(input.size(), input.type());
  for (int y = 0; y < output.rows; y++)
    for (int x = 0; x < output.cols; x++)
    {
      int label = labels.at<int>(y, x);
      CV_Assert(0 <= label && label <= nccomps);
      output.at<uchar>(y, x) = colors[label];
    }
}

int main(int argc, char** argv)
{

  if (argc < 2)
  {
    std::cout << "No map file was given!" << std::endl;
    return -1;
  }
  std::string map_file = argv[1];
  cv::Mat map_image =
      cv::imread(ROOT_SAMPLE + "/data/" + map_file, cv::IMREAD_UNCHANGED);
  std::cout << "Map file: " << map_file.c_str() << std::endl;
  if (map_image.empty())
  {
    return -2;
  }
  cv::imshow("0.Map image", map_image);

  /* Seperate obstacles, unknown and freespace areas */
  cv::Mat map_ob =
      cv::Mat::zeros(map_image.rows, map_image.cols, map_image.type());
  cv::Mat map_un =
      cv::Mat::zeros(map_image.rows, map_image.cols, map_image.type());
  cv::Mat map_fr =
      cv::Mat::zeros(map_image.rows, map_image.cols, map_image.type());

  for (int i = 0; i < map_image.rows; i++)
  {
    for (int j = 0; j < map_image.cols; j++)
    {
      uchar pixel_val = map_image.at<uchar>(i, j);
      if (pixel_val < 205)
        map_ob.at<uchar>(i, j) = 255;
      else if (pixel_val > 205)
        map_fr.at<uchar>(i, j) = 255;
      else
        map_un.at<uchar>(i, j) = 255;
    }
  }

  cv::Mat map_ob_filtered;
  remove_small_components(map_ob, map_ob_filtered, OBSTACLE_FILTER_THRESH);
  cv::imshow("map_ob_filtered", map_ob_filtered);

  /* Open freespace */
  int open_kernel_size_w = 1;
  cv::Mat open_element_w = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * open_kernel_size_w + 1, 2 * open_kernel_size_w + 1));

  cv::Mat map_fr_open;
  cv::morphologyEx(map_fr, map_fr_open, cv::MORPH_OPEN, open_element_w);
  cv::imshow("map_fr_open", map_fr_open);

  cv::Mat map_fr_filtered;
  remove_small_components(map_fr_open, map_fr_filtered,
                          FREESPACE_FILTER_THRESH);
  cv::imshow("map_fr_filtered", map_fr_filtered);

  /* Close freespace */
  int close_kernel_size_w = 3;
  cv::Mat close_element_w = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * close_kernel_size_w + 1, 2 * close_kernel_size_w + 1));

  cv::Mat map_fr_close;
  cv::morphologyEx(map_fr_filtered, map_fr_close, cv::MORPH_CLOSE,
                   close_element_w);
  cv::imshow("map_fr_close", map_fr_close);

  /* Combine 3 layers */
  cv::Mat map_combine = cv::Mat::zeros(map_image.size(), map_image.type());
  for (int i = 0; i < map_combine.rows; i++)
  {
    for (int j = 0; j < map_combine.cols; j++)
    {
      if (map_ob_filtered.at<uchar>(i, j) == 255)
        map_combine.at<uchar>(i, j) = 0;
      else if (map_fr_close.at<uchar>(i, j) == 255)
        map_combine.at<uchar>(i, j) = 255;
      else
        map_combine.at<uchar>(i, j) = 205;
    }
  }
  cv::imshow("map_combine", map_combine);

  /* Find freespace contours and draw on combined map */
  cv::Mat map_combine_with_contours = map_combine.clone();
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(map_fr_close, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  for (size_t i = 0; i < contours.size(); i++)
    cv::drawContours(map_combine_with_contours, contours, (int)i,
                     cv::Scalar(0, 0, 0), 1);

  cv::imshow("Combine with contours", map_combine_with_contours);

  cv::imwrite("/home/ros/" + map_file + "_b", map_combine_with_contours);

  cv::waitKey(0);

  return 0;
}
