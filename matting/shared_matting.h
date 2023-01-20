#ifndef SHAREDMSTTING_H
#define SHAREDMSTTING_H
//修复了内存溢出的问题
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>

struct labelPoint
{
    int x;
    int y;
    int label;
};


struct Tuple
{
    cv::Scalar f;
    cv::Scalar b;
    double   sigmaf;
    double   sigmab;

    int flag;

};

struct Ftuple
{
    cv::Scalar f;
    cv::Scalar b;
    double   alphar;
    double   confidence;
};

/*程序中认定cv::Point中 x为行，y为列，可能错误，但对程序结果没有影响*/
class SharedMatting
{
public:
    SharedMatting();
    ~SharedMatting();

    void loadImage(cv::Mat pImg);
    void loadTrimap(cv::Mat Trimap);
    void expandKnown();
    void sample(cv::Point p, std::vector<cv::Point>& f, std::vector<cv::Point>& b);
    void gathering();
    void refineSample();
    void localSmooth();
    void solveAlpha();
    void save(char* filename);
    void Sample(std::vector<std::vector<cv::Point> >& F, std::vector<std::vector<cv::Point> >& B);
    void matteProcess();
    //未定义该函数
    void release();

    double mP(int i, int j, cv::Scalar f, cv::Scalar b);
    double nP(int i, int j, cv::Scalar f, cv::Scalar b);
    double eP(int i1, int j1, int i2, int j2);
    double pfP(cv::Point p, std::vector<cv::Point>& f, std::vector<cv::Point>& b);
    double aP(int i, int j, double pf, cv::Scalar f, cv::Scalar b);
    double gP(cv::Point p, cv::Point fp, cv::Point bp, double pf);
    double gP(cv::Point p, cv::Point fp, cv::Point bp, double dpf, double pf);
    double dP(cv::Point s, cv::Point d);
    double sigma2(cv::Point p);
    double distanceColor2(cv::Scalar cs1, cv::Scalar cs2);
    double comalpha(cv::Scalar c, cv::Scalar f, cv::Scalar b);

    cv::Mat& getMatte() { return matte; }

private:
  //    IplImage * pImg;
  //    IplImage * trimap;
  //    IplImage * matte;
  cv::Mat pImg;
  cv::Mat trimap;

  cv::Mat matte;

    std::vector<cv::Point> uT;
    std::vector<struct Tuple> tuples;
    std::vector<struct Ftuple> ftuples;

    int height;
    int width;
    int kI;
    int kG;
    int** unknownIndex;//Unknown的索引信息；
    int** tri;
    int** alpha;
    double kC;
    
    int step;
    int channels;
    uchar* data;

};



#endif
