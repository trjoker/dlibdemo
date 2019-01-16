//
// Created by lvfei on 2018/11/6.
//

#ifndef DLIBDEMO_EYESHADOW_H
#define DLIBDEMO_EYESHADOW_H

#include <math.h>
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
//变形算法1，MLS，刚性变化
/*
class MLSD{
public:
    MLSD();
    virtual ~MLSD();

    Mat Initialize(const Mat &oriImg, const std::vector<cv::Point2d> &qsrc, const std::vector<Point2d> &qdst,
                   const int outW, const int outH, const int grid_size = 10, const double al_pha = 1.0);

    Mat genNewImg(const Mat &oriImg);

    virtual void calcDelta() = 0;           //need to implement by this son class

    void setSrcPoints(const std::vector<Point2d> &qsrc);
    void setDstPoints(const std::vector<Point2d> &qdst);
    void setSize(int w, int h){
        srcW = w;srcH = h;
    }
    void setOutSize(int w, int h){
        tarW = w; tarH = h;
    }

protected:
    double alpha;
    int gridsize;

    std::vector<Point2d> oldDotL, newDotL;             //old : srcPoint  new : dstPoint

    int nPoint;

    Mat_<double> rDx, rDy;          //deltax and deltay

    int srcW, srcH;
    int tarW, tarH;

};

//根据关键点移动所有坐标
class MLSDRigid : public MLSD
{
public:
    MLSDRigid() {}
    void calcDelta();
};
//变形算法1MLS结束
*/
//变形算法二，高效的MLS***************************************************************************
template <typename T>
const T& clamp(const T& value, const T& min, const T& max)
{
    assert(min <= max && "invalid clamp range");
#if 0
    return std::min<T>(std::max<T>(value, min), max);
#else
    if (value < min)  return min;
    if (value > max)  return max;
    return value;
#endif
}

template <typename T>
cv::Vec<T, 4> boundingBox(const std::vector<cv::Point_<T>>& points, int start, int length)
{
    assert(points.size() >= 3);

    const cv::Point_<T>& p0 = points[start];
    float left = p0.x, right = p0.x, top = p0.y, bottom = p0.y;
    for (int i = 1; i < length; ++i)
    {
        const cv::Point_<T>& point = points[start + i];
        if (point.x < left)
            left = point.x;
        else if (point.x > right)
            right = point.x;

        if (point.y < top)
            top = point.y;
        else if (point.y > bottom)
            bottom = point.y;
    }

    return cv::Vec<T, 4>(left, top, right, bottom);
}

template <typename T>
inline cv::Vec<T, 4> boundingBox(const std::vector<cv::Point_<T>>& points)
{
    return boundingBox(points, 0, static_cast<int>(points.size()));
}
class ImageWarp
{
private:

protected:
    int grid_size; ///< Parameter for MLS.

    std::vector<cv::Point2f> src_points;
    std::vector<cv::Point2f> dst_points;

    cv::Mat_<float> rDx, rDy;

    cv::Size2i src_size;
    cv::Size2i dst_size;

public:
    ImageWarp();
    ImageWarp(int grid_size);
    virtual ~ImageWarp(){}

    /**
     * Set all and generate an output.
     * @param src        The input image to be warped.
     * @param src_points A list of "from" points.
     * @param dst_points A list of "target" points.
     * @param target     The output image size.
     * @param amount     1 means warp to target points, 0 means no warping.
     *
     * This will do all the initialization and generate a warped image. After calling this, one can later call
     * genNewImage with different transRatios to generate a warping animation.
     */
    cv::Mat setAllAndGenerate(const cv::Mat& src,
                              const std::vector<cv::Point2f> &src_points, const std::vector<cv::Point2f> &dst_points,
                              const cv::Size2i& target, float alpha, float amount = 1.0F);

    /**
     * Generate the warped image.
     * This function generate a warped image using PRE-CALCULATED data.
     * DO NOT CALL THIS AT FIRST! Call this after at least one call of setAllAndGenerate.
     */
    cv::Mat genNewImage(const cv::Mat& src, float transRatio);

    /**
     * Calculate delta value which will be used for generating the warped image.
     */
    virtual void calculateDelta(float alpha) = 0;



    /**
     * @param[in] dst_points Set the list of target points
     * @param[in] src_points Set the list of source points
     */
    void setMappingPoints(const std::vector<cv::Point2f>& dst_points, const std::vector<cv::Point2f>& src_points);
//	void setMappingPoints(const std::vector<cv::Point2f>&& dst_points, const std::vector<cv::Point2f>&& src_points);

    void setSourceSize(int width, int height) { src_size = cv::Size2i(width, height); }
    void setTargetSize(int width, int height) { dst_size = cv::Size2i(width, height); }

    void setSourceSize(const cv::Size2i& size) { src_size = size; }
    void setTargetSize(const cv::Size2i& size) { dst_size = size; }
};

/**
 * The class for MLS Rigid transform.
 * It will try to keep the image rigid. You can set preScale if you
 * can accept uniform transform.
 */
class ImageWarp_Rigid : public ImageWarp
{
private:
    bool prescale;  ///< Whether unify scaling the points before deformation

public:
    ImageWarp_Rigid();
    virtual void calculateDelta(float alpha) override;

    void set(bool prescale);
};


//眼影对齐上色
void eyeshadow(cv::Mat src,std::vector<cv::Point> shape_points,cv::Mat tmp_left,cv::Mat tmp_right);


#endif //DLIBDEMO_EYESHADOW_H
