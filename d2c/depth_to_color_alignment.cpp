//本项目为2022实验室深度对齐算法 作者：潘颢文 修改：秦禹康

#include"depth_to_color_alignment.h"

#define D2C_SOLUTION 1


D2C::D2C(int Width, int Height) {
    read_depth_color(Width, Height);
}

//读取图片信息
void D2C::read_depth_color(int Width, int Height)
{
    this->width = Width;
    this->height = Height;
}

//读取相机信息
void D2C::read_camera_data(std::string extrinsics_path, std::string intrinsics_path)

{

    //读取内外参
    extrinsics.open(extrinsics_path, cv::FileStorage::READ);
    if (extrinsics.isOpened())
    {
        extrinsics["R"] >> R;
        extrinsics["T"] >> T;
        std::cout << R << std::endl;
        extrinsics.release();
        std::cout << "READ extrinsics!" << std::endl;
    }



    intrinsics.open(intrinsics_path, cv::FileStorage::READ);
    if (intrinsics.isOpened())
    {
        intrinsics["M1"] >> cameraMatrix[0];
        intrinsics["D1"] >> distCoeffs[0];
        intrinsics["M2"] >> cameraMatrix[1];
        intrinsics["D2"] >> distCoeffs[1];
        intrinsics.release();
        std::cout << "READ intrinsics!" << std::endl;

    }


    std::cout << cameraMatrix[0] << std::endl;
    
}




// phw's feat
void D2C::set_rotate_map()
{
    cv::Mat depth_point = cv::Mat::zeros(cv::Size(width, height), CV_32FC3);
    //cv::Mat ir_point = cv::Mat::zeros(cv::Size(ir_width, ir_height), CV_32FC3);

    // Attention, their names were written the opposite
    cv::Point2f center_rgb = cv::Point2f(cameraMatrix[1].at<double>(0, 2), cameraMatrix[1].at<double>(1, 2));
    cv::Point2f center = cv::Point2f(cameraMatrix[0].at<double>(0, 2), cameraMatrix[0].at<double>(1, 2));
    cv::Point2f foc_rgb = cv::Point2f(cameraMatrix[1].at<double>(0, 0), cameraMatrix[1].at<double>(1, 1));
    cv::Point2f foc = cv::Point2f(cameraMatrix[0].at<double>(0, 0), cameraMatrix[0].at<double>(1, 1));

    std::cout << "center" << center_rgb << " " << center << std::endl;
    std::cout << "focus" << foc_rgb << " " << foc << std::endl;


    static cv::Mat oneMat = cv::Mat::ones(height, width, CV_16U) * 1000;

    cv::Mat R_inv = R.inv();
    //cv::Mat R_inv = R;            
     
    for (int i = 0; i < height; i++)
    {
        uint16_t* ptr = oneMat.ptr<uint16_t>(i);
        float* dp_ptr = depth_point.ptr<float>(i);
        for (int j = 0; j < width; j++) {

            // 第一个点
            cv::Point3d newPoint;
            //像素转换至点云
            newPoint.x = (j - center.x) * *ptr / foc.x;
            newPoint.y = (i - center.y) * *ptr / foc.y;
            newPoint.z = *ptr;

            double* R_ptr = R_inv.ptr<double>(0);
            double nx, ny, nz;
            double* Tptr = T.ptr<double>(0);

            //nx = R.at<double>(0, 0) * newPoint.x + R.at<double>(0, 1) * newPoint.y + R.at<double>(0, 2) * newPoint.z - 12.61372;
            nx = *R_ptr++ * newPoint.x + *Tptr++;
            //nx = *R_ptr++ * newPoint.x - *Tptr++;
            nx += *R_ptr++ * newPoint.y;
            nx += *R_ptr++ * newPoint.z;

            //ny = R.at<double>(1, 0) * newPoint.x + R.at<double>(1, 1) * newPoint.y + R.at<double>(1, 2) * newPoint.z - -1.494398;
            R_ptr = R.ptr<double>(1);
            ny = *R_ptr++ * newPoint.x + *Tptr++;
            //ny = *R_ptr++ * newPoint.x - *Tptr++;
            ny += *R_ptr++ * newPoint.y;
            ny += *R_ptr++ * newPoint.z;

            //nz = R.at<double>(2, 0) * newPoint.x + R.at<double>(2, 1) * newPoint.y + R.at<double>(2, 2) * newPoint.z - 3.078241999;
            R_ptr = R.ptr<double>(2);
            nz = *R_ptr++ * newPoint.x + *Tptr++;
            //nz = *R_ptr++ * newPoint.x - *Tptr++;
            nz += *R_ptr++ * newPoint.y;
            nz += *R_ptr++ * newPoint.z;

            //depth点云赋值,点云转换回像素
            *dp_ptr++ = nx / *ptr * foc_rgb.x + center_rgb.x;
            *dp_ptr++ = ny / *ptr * foc_rgb.y + center_rgb.y;
            *dp_ptr++ = nz;
            ptr++;
        }

    }
    std::vector<cv::Mat> channels;
    cv::split(depth_point, channels);
    X_map = channels[0];
    Y_map = channels[1];
    //d2c_remap(X_map, Y_map, rotate_depth);
    //d2c_remap(X_map, Y_map, rotate_ir);
}

void D2C::d2c_remap(cv::Mat& inputImage,cv::Mat& outputImage) {
    cv::remap(inputImage, outputImage, X_map, Y_map, cv::INTER_LINEAR);
}