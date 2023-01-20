///*本项目为2022实验室D2C配准SDK*/
///*作者：潘颢文、秦禹康(2022.08.19修改)*/
/*本项目为2022实验室D2C配准SDK*/
/*作者：潘颢文、秦禹康(2022.08.19修改)*/
//知道啦，不用一直重复的
#pragma once
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fstream>
#include <ctime>



class D2C {
public:
	D2C(int Width, int Height);

	void read_depth_color(int Width, int Height);

	void read_camera_data(std::string extrinsics_path, std::string intrinsics);

	// 手动传是防止没有初始化XYMap
	void d2c_remap(cv::Mat& inputImage, cv::Mat& outputImage);

	void set_rotate_map();


private:
	cv::Mat X_map, Y_map;

	cv::FileStorage extrinsics, intrinsics, rect;

	int height, width;

	cv::Mat R, T;

	cv::Mat cameraMatrix[2], distCoeffs[2];

};

