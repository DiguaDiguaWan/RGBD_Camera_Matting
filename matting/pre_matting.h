#ifndef PRE_MATTING_H
#define PRE_MATTING_H

#include <iostream>
#include <cmath>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class trimap_rgb_dep {

 public:
  trimap_rgb_dep();
  ~trimap_rgb_dep();

  void resize_min();
  void trimap_process();
  void read_info(cv::Mat img, cv::Mat dep);
  void dep_gating(int max, int min);
  void largest_connected_component(cv::Mat src, cv::Mat &dst, int maximum);
  void mix_last_matte(cv::Mat &matte);
  void mix_dep(cv::Mat matte);

  cv::Mat get_image() const { return m_image; }
  cv::Mat get_dep_g() const { return m_dep_g; }
  cv::Mat get_image_min() const { return m_image_min; }
  cv::Mat get_trimap() const { return m_trimap; }
  cv::Mat get_trimap_min() const { return m_trimap_min; }

 private:
  void find_hair_pixel(cv::Mat &hair);

  // 类的成员变量 m_
  // static k_
  // 局部 p_
  // 全局 g_
  // 全局静态 g_k_

 private:
  cv::Mat m_image;
  cv::Mat m_image_min;
  cv::Mat m_trimap;
  cv::Mat m_trimap_min;

  cv::Mat m_foreground;
  cv::Mat m_background;
  cv::Mat m_last_matte;
  cv::Mat m_dep_g;
  cv::Mat m_dep;

  int m_img_height;
  int m_img_width;
  int m_img_channels;
  int m_dep_height;
  int m_dep_width;
  int m_dep_channels;
  int m_dep_max_value;
  int m_dep_min_value;

  long int m_dep_g_all_pixel;

};

struct cc{
  cv::Mat image;
  cv::Mat image_min;
  cv::Mat trimap;
  cv::Mat trimap_min;
};

#endif


