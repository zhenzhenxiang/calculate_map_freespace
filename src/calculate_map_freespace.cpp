#include <opencv2/opencv.hpp>
#include <iostream>

const std::string ROOT_SAMPLE = "/home/ros/Projects/Calculate_map_freespace";
const int THRESHOLD = 210;
const int UNKNOWN = 200;
const float RESOLUTION = 0.02;
const int CANNY_THRESHOLD_UP = 100;
const int CANNY_THRESHOLD_DOWN = 200;
float SCALE = 0.6;
float INNER_AREA_THRESH_M = 3.0; // m*m
float INNER_AREA_THRESH =
    INNER_AREA_THRESH_M / RESOLUTION / RESOLUTION; // pixel*pixel

cv::RNG rng(12345);

int main(int argc, char **argv) {

  if (argc < 3) {
    std::cout << "No map file or floor height was given!" << std::endl;
    return -1;
  }
  std::string map_file = argv[1];
  cv::Mat map_image =
      cv::imread(ROOT_SAMPLE + "/data/" + map_file, cv::IMREAD_UNCHANGED);
  std::cout << "Map file: " << map_file.c_str() << std::endl;
  if (map_image.empty()) {
    return -2;
  }
  cv::imshow("0.Map image", map_image);

  std::string floor_height_string = argv[2];
  float floor_height = std::stof(floor_height_string, 0);
  std::cout << "Floor height was set to: " << floor_height << std::endl;

  cv::Mat map_mB;
  cv::medianBlur(map_image, map_mB, 5);
  cv::imshow("1.Median blur", map_mB);

  /* Segment unknown area */
  cv::Mat map_thresh;
  cv::inRange(map_mB, UNKNOWN, THRESHOLD, map_thresh);
  map_thresh = 255 - map_thresh;
  cv::imshow("2.Threshold unknown", map_thresh);

  /* Remove seperate white points by morphology operation: open */
  int open_kernel_size_w = 1; // 5
  cv::Mat open_element_w = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * open_kernel_size_w + 1, 2 * open_kernel_size_w + 1));

  cv::Mat map_image_open_w;
  cv::morphologyEx(map_thresh, map_image_open_w, cv::MORPH_OPEN,
                   open_element_w);
  cv::imshow("3.Map with white open", map_image_open_w);

  /* Fill white holes by morphology operation: close */
  int close_kernel_size_th_w = 3; // 10
  cv::Mat close_element_th_w = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(2 * close_kernel_size_th_w + 1, 2 * close_kernel_size_th_w + 1));

  cv::Mat map_thresh_close;
  cv::morphologyEx(map_image_open_w, map_thresh_close, cv::MORPH_CLOSE,
                   close_element_th_w);
  cv::imshow("4.Map with white close", map_thresh_close);

  /* Find contour */
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(map_thresh_close, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

  cv::Mat map_free = cv::Mat::zeros(map_thresh_close.size(), CV_8UC3);
  cv::Mat map_contours = map_free.clone();
  double max_area_pixel = 0.0;
  int largest_contour_ind = 0;
  for (size_t i = 0; i < contours.size(); i++) {
    double area = cv::contourArea(contours[i]);

    if (area > max_area_pixel) {
      largest_contour_ind = i;
      max_area_pixel = area;
    }

    cv::drawContours(map_contours, contours, (int)i,
                     cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                rng.uniform(0, 255)),
                     2);
  }
  cv::imshow("5.Contours", map_contours);

  /* Draw the largest contour */
  cv::Scalar color = cv::Scalar(255, 0, 255);
  cv::drawContours(map_free, contours, (int)largest_contour_ind, color, 2);

  /* Fit the largest contour with polylines and fill it to calculate area*/
  std::vector<cv::Point> approx;
  std::vector<std::vector<cv::Point>> approx_vector;
  cv::approxPolyDP(contours[largest_contour_ind], approx, 3, true);
  approx_vector.push_back(approx);
  cv::fillPoly(map_free, approx_vector, cv::Scalar(0, 255, 0));
  cv::imshow("6.Freespace", map_free);

  /* Remove non-accessible area */
  std::vector<std::vector<cv::Point>> inner_contours = contours;
  std::cout << "inner_contours size: " << inner_contours.size() << std::endl;
  inner_contours.erase(inner_contours.begin() + largest_contour_ind);
  std::cout << "inner_contours size: " << inner_contours.size() << std::endl;

  for (size_t i = 0; i < inner_contours.size(); i++) {
    double area = cv::contourArea(inner_contours[i]);
    if (area > INNER_AREA_THRESH)
      cv::drawContours(map_free, inner_contours, (int)i, cv::Scalar(0, 0, 0),
                       -1);
    cv::drawContours(map_free, inner_contours, (int)i, cv::Scalar(255, 0, 255),
                     1);
  }
  cv::imshow("7.Freespace without holes", map_free);

  cv::Mat map_color, map_merged;
  cv::cvtColor(map_image, map_color, CV_GRAY2BGR);
  cv::addWeighted(map_color, 0.85, map_free, 0.15, 0, map_merged);
  cv::polylines(map_merged, approx, true, cv::Scalar(0, 0, 255), 1);

  /* Calculate floor volume */
  long int floor_area_pixel = 0;
  for (int j = 0; j < map_free.rows; j++) {
    for (int i = 0; i < map_free.cols; i++) {
      cv::Vec3b intensity = map_free.at<cv::Vec3b>(j, i);
      if (intensity.val[1] == 255)
        floor_area_pixel++;
    }
  }

  //  double floor_area = cv::contourArea(approx) * RESOLUTION * RESOLUTION;
  double floor_area = floor_area_pixel * RESOLUTION * RESOLUTION;
  double floor_volume = floor_area * floor_height;
  std::cout << "Floor area: " << floor_area
            << ", floor volume: " << floor_volume << std::endl;

  std::string floor_height_txt =
      "Height: " + std::to_string(floor_height) + "m";
  std::string floor_area_txt = "Area: " + std::to_string(floor_area) + "m2";
  std::string floor_volume_txt =
      "Volume: " + std::to_string(floor_volume) + "m3";

  cv::putText(map_merged, floor_height_txt, cv::Point(30, 100),
              cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2);
  cv::putText(map_merged, floor_area_txt, cv::Point(30, 200),
              cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2);
  cv::putText(map_merged, floor_volume_txt, cv::Point(30, 300),
              cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255), 2);

  /* Resize to show */
  cv::resize(map_merged, map_merged,
             cv::Size(map_merged.cols * SCALE, map_merged.rows * SCALE));

  cv::imshow("8.Merged", map_merged);

  cv::waitKey(0);

  return 0;
}
