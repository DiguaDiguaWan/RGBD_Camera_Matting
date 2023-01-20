//基于深度生成trimap代码
//create by qinyukang@2022laboratory.

#include "pre_matting.h"
#include <opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <stdio.h>
#include <time.h>

struct mat_info
{
    cv::Mat dep;
};

//构建函数
trimap_rgb_dep::trimap_rgb_dep() {
  m_dep_g_all_pixel = 0;
}

//析构函数
trimap_rgb_dep::~trimap_rgb_dep() {
  //释放内存
  m_trimap.release();
  m_dep.release();
  m_image.release();
  m_dep_g.release();
  m_last_matte.release();;
  m_image_min.release();
  m_trimap_min.release();

}

//读取图片信息
void trimap_rgb_dep::read_info(cv::Mat img, cv::Mat dep) {

  m_dep = dep;
  m_dep_height = dep.rows;
  m_dep_width = dep.cols;
  m_dep_channels = dep.channels();

  m_image = img;
  m_img_height = img.rows;
  m_img_width = img.cols;
  m_img_channels = img.channels();

}

//深度选通
void trimap_rgb_dep::dep_gating(int max, int min) {
  m_dep_g = cv::Mat::zeros(cv::Size(m_dep_width, m_dep_height), CV_8UC1);
  m_dep_max_value = max;
  m_dep_min_value = min;

  for (int i = 0; i < m_dep_height; ++i) {
      auto * _dep_ptr=m_dep.ptr<uint16_t>(i);
      auto * _dep_g_ptr=m_dep_g.ptr<uint8_t>(i);
      for (int j = 0; j < m_dep_width; ++j) {

          auto _dep_info = *_dep_ptr;
          if (_dep_info < max && _dep_info > min) {
              *_dep_g_ptr = 255;
             m_dep_g_all_pixel += 1;
      }
          ++_dep_g_ptr;
          ++_dep_ptr;
    }
  }
  //消除飞点

  largest_connected_component(m_dep_g, m_dep_g, 1000);

}

//trimap过程
void trimap_rgb_dep::trimap_process() {
  //做一下膨胀：
  cv::Mat dep_foreground;
  cv::Mat dep_find_area;
  cv::Mat dep_mix_area;

  cv::Mat element_1 = getStructuringElement(cv::MORPH_RECT, cv::Size(80, 80));
  cv::Mat element_2 = getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
  cv::Mat element_3 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::Mat element_4 = getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
  cv::Mat element_5 = getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));

  cv::erode(m_dep_g, dep_foreground, element_2);
  cv::dilate(m_dep_g, dep_find_area, element_1);
  cv::dilate(m_dep_g, dep_mix_area, element_3);
  dep_mix_area -= dep_foreground;

  m_background = cv::Mat(m_dep_height, m_dep_width, CV_8UC1);
  m_foreground = dep_foreground.clone();
  m_trimap = cv::Mat::zeros(m_dep_height, m_dep_width, CV_8UC3);

  cv::Mat hair = cv::Mat::zeros(cv::Size(m_dep_width, m_dep_height), CV_8UC1);

  for (int i = 0; i < m_dep_height; ++i) {

    for (int j = 0; j < m_dep_width; ++j) {

      auto bgr = *(m_image.ptr<cv::Vec3b>(i)+j);
      auto Cb = 128 - 37.797 * bgr[2] / 255 - 74.203 * bgr[1] / 255 + 112 * bgr[0] / 255;
      auto Cr = 128 + 112 * bgr[2] / 255 - 93.768 * bgr[1] / 255 - 18.214 * bgr[0] / 255;

      if (*(dep_mix_area.ptr<uint8_t>(i)+j)> 0 && (Cb >= 115 && Cb <= 141 && Cr >= 115 && Cr <= 143 && bgr[2] < 100)) {
        hair.at<uint8_t>(i, j) = 255;
      }

    }
  }

  largest_connected_component(hair, hair, 50);
  find_hair_pixel(hair);
  hair += dep_foreground;
  erode(hair, hair, element_4);
  largest_connected_component(hair, hair, 10000);
  dilate(hair, hair, element_5);

  for (int i = 0; i < m_dep_height; ++i) {
    for (int j = 0; j < m_dep_width; ++j) {
      auto hair_info = *(hair.ptr<uint8_t>(i)+j);
      if (hair_info > 0) {
        m_trimap.at<cv::Vec3b>(i, j) = cv::Vec3b(125, 125, 125);
        if (m_foreground.at<uint8_t>(i, j) == 255) {
          m_trimap.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
        } else {
          uint16_t dep_info = *(m_dep.ptr<uint16_t>(i)+j);
          auto bgr = *(m_image.ptr<cv::Vec3b>(i)+j);
          auto Cb = 128 - 37.797 * bgr[2] / 255 - 74.203 * bgr[1] / 255 + 112 * bgr[0] / 255;
          auto Cr = 128 + 112 * bgr[2] / 255 - 93.768 * bgr[1] / 255 - 18.214 * bgr[0] / 255;
          if (dep_info == 0 && (Cb >= 115 && Cb <= 141 && Cr >= 115 && Cr <= 143 && bgr[2] < 80)) {
              *(m_trimap.ptr<cv::Vec3b>(i)+ j) = cv::Vec3b(255, 255, 255);
          }
        }

      }

    }
  }

}

//保留最大连通区域
void trimap_rgb_dep::largest_connected_component(cv::Mat src, cv::Mat &dst, int maximum) {
  cv::Mat temp;
  cv::Mat labels;
  src.copyTo(temp);
  //标记连通域
  int n_comps = connectedComponents(temp, labels, 8, CV_16U);
  std::vector<int> histogram_of_labels;
  for (int i = 0; i < n_comps; i++)//初始化labels的个数为0
  {
    histogram_of_labels.push_back(0);
  }

  int rows = labels.rows;
  int cols = labels.cols;
  for (int row = 0; row < rows; row++) //计算每个labels的个数
  {
    for (int col = 0; col < cols; col++) {
      histogram_of_labels.at(labels.at<uint16_t>(row, col)) += 1;
    }
  }
  histogram_of_labels.at(0) = 0; //将背景的labels个数设置为0

  //计算最大的连通域labels索引

  //int max_idx = 0;
  std::vector<int> max_idx;
  for (int i = 0; i < n_comps; i++) {    //判断面积是否大于阈值
    if (histogram_of_labels.at(i) > maximum) {
      //maximum = histogram_of_labels.at(i);
      max_idx.push_back(i);
    }
  }

  //将最大连通域标记
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      std::vector<int>::iterator t;
      t = find(max_idx.begin(), max_idx.end(), labels.at<uint16_t>(row, col));

      if (t != max_idx.end()) {
        labels.at<uint16_t>(row, col) = 255;
      } else {
        labels.at<uint16_t>(row, col) = 0;
      }
    }
  }
  //将图像更改为CV_8U格式
  labels.convertTo(dst, CV_8U);

}

//寻找头发区域像素
void trimap_rgb_dep::find_hair_pixel(cv::Mat &hair) {
  cv::Mat label = cv::Mat::zeros(m_dep_height, m_dep_width, CV_8UC1);
  for (int i = 0; i < m_dep_height; ++i) {
    for (int j = 0; j < m_dep_width; ++j) {

      auto hair_info = *(hair.ptr<uint8_t>(i)+j);
      if (hair_info == 255) {
        bool iterator_hair = true;
        int n = 0;
        while (iterator_hair) {
          if (i <= n) iterator_hair = false;
          if (label.at<uint8_t>(i - n, j) == 255)iterator_hair = false;
          auto dep_info = m_dep.at<uint16_t>(i - n, j);
          auto bgr = m_image.at<cv::Vec3b>(i - n, j);
          auto Cb = 128 - 37.797 * bgr[2] / 255 - 74.203 * bgr[1] / 255 + 112 * bgr[0] / 255;
          auto Cr = 128 + 112 * bgr[2] / 255 - 93.768 * bgr[1] / 255 - 18.214 * bgr[0] / 255;
          if (dep_info == 0 && (Cb >= 115 && Cb <= 141 && Cr >= 115 && Cr <= 143 && bgr[2] < 100)) {
            //标记头发区域像素
            label.at<uint8_t>(i - n, j) = 255;
          } else { iterator_hair = false; }
          n++;
        }

      }

    }
  }
  hair += label;
}

//考虑上一帧matte
void trimap_rgb_dep::mix_last_matte(cv::Mat &matte) {
  for (int i = 0; i < matte.rows; ++i) {
    for (int j = 0; j < matte.cols; ++j) {
      auto matte_info = *(matte.ptr<uint8_t>(i)+j);
      if (matte_info == 0 && *(m_trimap.ptr<cv::Vec3b>(i)+j) == cv::Vec3b(255, 255, 255)) {
        auto bgr = *(m_image.ptr<cv::Vec3b>(i)+j);
        auto Cb = 128 - 37.797 * bgr[2] / 255 - 74.203 * bgr[1] / 255 + 112 * bgr[0] / 255;
        auto Cr = 128 + 112 * bgr[2] / 255 - 93.768 * bgr[1] / 255 - 18.214 * bgr[0] / 255;
        if (Cb >= 115 && Cb <= 141 && Cr >= 115 && Cr <= 143 && bgr[2] < 100) {
            *(m_trimap.ptr<cv::Vec3b>(i)+j) = cv::Vec3b(125, 125, 125);
        }

      }
      if (matte_info >= 0 && *(m_trimap.ptr<cv::Vec3b>(i)+j) == cv::Vec3b(125, 125, 125)) {
          auto bgr = *(m_image.ptr<cv::Vec3b>(i)+j);
        auto Cb = 128 - 37.797 * bgr[2] / 255 - 74.203 * bgr[1] / 255 + 112 * bgr[0] / 255;
        auto Cr = 128 + 112 * bgr[2] / 255 - 93.768 * bgr[1] / 255 - 18.214 * bgr[0] / 255;
        if (Cb >= 115 && Cb <= 141 && Cr >= 115 && Cr <= 143 && bgr[2] < 60) {
            *(m_trimap.ptr<cv::Vec3b>(i)+j) = cv::Vec3b(255, 255, 255);
        }

      }
      if (matte_info == 0 && *(m_trimap.ptr<cv::Vec3b>(i)+j) == cv::Vec3b(125, 125, 125)) {
          auto bgr = *(m_image.ptr<cv::Vec3b>(i)+j);
        auto Cb = 128 - 37.797 * bgr[2] / 255 - 74.203 * bgr[1] / 255 + 112 * bgr[0] / 255;
        auto Cr = 128 + 112 * bgr[2] / 255 - 93.768 * bgr[1] / 255 - 18.214 * bgr[0] / 255;
        if (!(Cb >= 115 && Cb <= 141 && Cr >= 115 && Cr <= 143 && bgr[2] < 80)) {
            *(m_trimap.ptr<cv::Vec3b>(i)+j) = cv::Vec3b(0, 0, 0);
        }

      }

    }

  }

}

//混叠深度
void trimap_rgb_dep::mix_dep(cv::Mat matte) {
  for (int i = 0; i < m_dep_height; ++i) {
    for (int j = 0; j < m_dep_width; ++j) {
      if (*(m_dep_g.ptr<uint8_t>(i)+j) > 0) {
          *(matte.ptr<uchar>(i)+j) = 255;
      }
    }
  }

}

//变换尺寸
void trimap_rgb_dep::resize_min() {
  cv::resize(m_image, m_image_min, cv::Size(int(m_img_width / 2), int(m_img_height / 2)), cv::INTER_AREA);
  cv::resize(m_trimap, m_trimap_min, cv::Size(int(m_img_width / 2), int(m_img_height / 2)), cv::INTER_AREA);
}
