//
// Created by lvfei on 2018/11/6.
//

#include "eyeshadow.h"
//变形算法1，MLS
/*
MLSD::MLSD(){

}

MLSD::~MLSD(){

}

inline double bilinear_interp(double x, double y, double v11, double v12,
                              double v21, double v22) {
    return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x;
}

Mat MLSD::Initialize(const Mat &oriImg, const std::vector<Point2d> &qsrc, const std::vector<Point2d> &qdst,
                     const int outW, const int outH, const int grid_size, const double al_pha){
    gridsize = grid_size;
    alpha = al_pha;

    setSize(oriImg.cols, oriImg.rows);
    setOutSize(outW, outH);

    setSrcPoints(qsrc);
    setDstPoints(qdst);

    // cout<<"done"<<endl;    //just used for test
    calcDelta();                //get rDx and rDy to apply bilinear interplation
    return genNewImg(oriImg);
}

void MLSD::setSrcPoints(const std::vector<Point2d> &qsrc){
    nPoint = int(qsrc.size());
    oldDotL.clear();
    for(int i = 0; i < nPoint; i++)
        oldDotL.push_back(qsrc[i]);         //reverse, but has no influence
}

void MLSD::setDstPoints(const std::vector<Point2d> &qdst){
    nPoint = int(qdst.size());
    newDotL.clear();
    for(int i = 0; i < nPoint; i++)
        newDotL.push_back(qdst[i]);
}

Mat MLSD::genNewImg(const Mat &oriImg) {
    int i, j;
    double di, dj;
    double nx, ny;
    int nxi, nyi, nxi1, nyi1;
    double deltaX, deltaY;
    double w, h;
    int ni, nj;

    Mat newImg(tarH, tarW, oriImg.type());
    for (i = 0; i < tarH; i += gridsize)                //i : y
        for (j = 0; j < tarW; j += gridsize) {         //j : x
            ni = i + gridsize, nj = j + gridsize;
            w = h = gridsize;
            if (ni >= tarH)     ni = tarH - 1, h = ni - i + 1;
            if (nj >= tarW)     nj = tarW - 1, w = nj - j + 1;
            for (di = 0; di < h; di++)
                for (dj = 0; dj < w; dj++) {
                    deltaX =
                            bilinear_interp(di / h, dj / w, rDx(i, j), rDx(i, nj),
                                            rDx(ni, j), rDx(ni, nj));
                    deltaY =
                            bilinear_interp(di / h, dj / w, rDy(i, j), rDy(i, nj),
                                            rDy(ni, j), rDy(ni, nj));
                    nx = j + dj + deltaX;
                    ny = i + di + deltaY;
                    if (nx > srcW - 1)      nx = srcW - 1;
                    if (ny > srcH - 1)       ny = srcH - 1;
                    if (nx < 0)     nx = 0;
                    if (ny < 0)     ny = 0;
                    nxi = int(nx);
                    nyi = int(ny);
                    nxi1 = ceil(nx);
                    nyi1 = ceil(ny);

                    if (oriImg.channels() == 1)
                        newImg.at<uchar>(i + di, j + dj) = bilinear_interp(
                                ny - nyi, nx - nxi, oriImg.at<uchar>(nyi, nxi),
                                oriImg.at<uchar>(nyi, nxi1),
                                oriImg.at<uchar>(nyi1, nxi),
                                oriImg.at<uchar>(nyi1, nxi1));
                    else {
                        for (int ll = 0; ll < 3; ll++)//just modify this for 4d,not 3d，need to add 3d  item
                            newImg.at<Vec3b>(i + di, j + dj)[ll] =
                                    bilinear_interp(
                                            ny - nyi, nx - nxi,
                                            oriImg.at<Vec3b>(nyi, nxi)[ll],
                                            oriImg.at<Vec3b>(nyi, nxi1)[ll],
                                            oriImg.at<Vec3b>(nyi1, nxi)[ll],
                                            oriImg.at<Vec3b>(nyi1, nxi1)[ll]);
                    }
                }
        }

//    for(int k = 0; k < nPoint; k++){//just test
//        circle( newImg, oldDotL[k], 3 , CV_RGB(0, 255, 0), FILLED, LINE_8, 0 );
//    }
//
//    for(int k = 0; k< nPoint; k++){
//        circle( newImg, newDotL[k], 3 , CV_RGB(0, 0, 255), FILLED, LINE_8, 0 );
//    }

    return newImg;
}

//关键点移动
void MLSDRigid::calcDelta(){
    int i, j, k;
    Point2d pstar, qstar;
    double sw = 0.0, ur = 0.0;
    double *w = new double[nPoint];

    rDx = rDx.zeros(tarH, tarW); rDy = rDy.zeros(tarH, tarW);       //alloc space

    for(i = 0; ; i += gridsize){                            //i : y
        if(i >= tarH && i < tarH + gridsize - 1)
            i = tarH - 1;
        else if(i >= tarH)
            break;
        for(j = 0; ; j += gridsize){                        //j : x
            if(j >= tarW && j < tarW + gridsize - 1)
                j = tarW - 1;
            else if(j >= tarW)
                break;

            pstar.x = pstar.y = 0;
            qstar.x = qstar.y = 0;

            for(k = 0; k < nPoint; k++){
                if((j == oldDotL[k].x) && (i == oldDotL[k].y))
                    continue;

                w[k] = 1 / pow(sqrt(pow(j-oldDotL[k].x, 2.0)+pow(i-oldDotL[k].y, 2.0)), 2 * alpha);
                sw += w[k];
                pstar += w[k] * oldDotL[k];
                qstar += w[k] * newDotL[k];
            }

            pstar /= sw;
            qstar /= sw;

            Mat M = Mat::zeros(2, 2, CV_64FC1);
            Mat P(1, 2, CV_64FC1), Q(1, 2, CV_64FC1);		//pi = p - pstar, qi = q - qstar
            Mat W = Mat::zeros(2, 2, CV_64FC1);
            double upq = 0, uqp = 0;

            for(k = 0; k < nPoint; k++){
                if (j == oldDotL[k].x && i == oldDotL[k].y)
                    continue;
                P.at<double>(0) = oldDotL[k].x - pstar.x;
                P.at<double>(1) = oldDotL[k].y - pstar.y;
                Q.at<double>(0) = newDotL[k].x - qstar.x;
                Q.at<double>(1) = newDotL[k].y- qstar.y;

                Mat DP(2, 2, CV_64FC1);
                DP.at<double>(0, 0) = -P.at<double>(0);
                DP.at<double>(0, 1) = P.at<double>(1);
                DP.at<double>(1, 0) = P.at<double>(1);
                DP.at<double>(1, 1) = P.at<double>(0);
                Mat DQ(2, 2, CV_64FC1);
                DQ.at<double>(0, 0) = -Q.at<double>(0);
                DQ.at<double>(0, 1) = Q.at<double>(1);
                DQ.at<double>(1, 0) = Q.at<double>(1);
                DQ.at<double>(1, 1) = Q.at<double>(0);

                upq += w[k] * Q.dot(P);
                P.at<double>(0) = -(oldDotL[k].y - pstar.y);
                P.at<double>(1) = oldDotL[k].x - pstar.x;
                uqp += w[k] * Q.dot(P);
                W += w[k] * (DP * DQ);
            }

            ur = sqrt(pow(upq, 2.0) + pow(uqp, 2.0));
            M = W / ur;
            ur = 0.0;
            Mat V(1, 2, CV_64FC1);
            V.at<double>(0) = j - pstar.x;
            V.at<double>(1) = i - pstar.y;
            Mat R = V * M;

            rDx.at<double>(i, j) = R.at<double>(0) + qstar.x - j;
            rDy.at<double>(i, j) = R.at<double>(1) + qstar.y - i;
        }
    }
}
//变形算法1，MLS结束
*/
//加入效率较高的变形算法二，MLS********************************************************************
ImageWarp::ImageWarp(int grid_size):
        grid_size(grid_size)
{
}

ImageWarp::ImageWarp():
        ImageWarp(10)
{
}

inline float bilinear_interp(float x, float y, float v11, float v12, float v21, float v22)
{
    return (v11*(1-y) + v12*y) * (1-x) + (v21*(1-y) + v22*y) * x;
}

cv::Mat ImageWarp::setAllAndGenerate(const cv::Mat& src,
                                     const std::vector<Point2f>& src_points,
                                     const std::vector<Point2f>& dst_points,
                                     const cv::Size& target, float alpha, float amount/* = 1.0F */)
{
    setSourceSize(src.cols, src.rows);
    setTargetSize(target);
    setMappingPoints(dst_points, src_points);
    calculateDelta(alpha);
    return genNewImage(src, amount);
}

cv::Mat ImageWarp::genNewImage(const cv::Mat& src, float amount)
{
    Mat dst(dst_size.height, dst_size.width, src.type());
    for(int i = 0; i < dst_size.height; i += grid_size)
        for(int j = 0; j < dst_size.width; j += grid_size)
        {
            int ni = i + grid_size;
            int nj = j + grid_size;
            float w = static_cast<float>(grid_size), h = w;
            if(ni >= dst_size.height) ni = dst_size.height - 1, h = static_cast<float>(ni - i + 1);
            if(nj >= dst_size.width)  nj = dst_size.width - 1,  w = static_cast<float>(nj - j + 1);

            for(int di = 0; di < h; ++di)
                for(int dj = 0; dj < w; ++dj)
                {
                    float deltaX = bilinear_interp(di/h, dj/w, rDx(i,j),rDx(i, nj), rDx(ni, j), rDx(ni, nj));
                    float deltaY = bilinear_interp(di/h, dj/w, rDy(i,j),rDy(i, nj), rDy(ni, j), rDy(ni, nj));
                    float nx = j + dj + deltaX * amount;
                    float ny = i + di + deltaY * amount;

                    nx = clamp(nx, 0.0F, static_cast<float>(src_size.width - 1));
                    ny = clamp(ny, 0.0F, static_cast<float>(src_size.height - 1));

                    int x0 = int(nx), y0 = int(ny);
                    int x1 = std::ceil(nx), y1 = std::ceil(ny);

                    switch(src.channels())
                    {
                        case 1:
                            dst.at<uchar>(i + di, j + dj) = bilinear_interp(ny - y0, nx - x0,
                                                                            src.at<uchar>(y0, x0), src.at<uchar>(y0, x1),
                                                                            src.at<uchar>(y1, x0), src.at<uchar>(y1, x1));
                            break;
                        case 3:
                            for(int b=0; b<3; ++b)
                                dst.at<Vec3b>(i + di, j + dj)[b] = bilinear_interp(ny - y0, nx - x0,
                                                                                   src.at<Vec3b>(y0, x0)[b], src.at<Vec3b>(y0, x1)[b],
                                                                                   src.at<Vec3b>(y1, x0)[b], src.at<Vec3b>(y1, x1)[b]);
                            break;
                        case 4:
                        {
                            for(int b=0; b<4; ++b)
                                dst.at<Vec4b>(i + di, j + dj)[b] = bilinear_interp(ny - y0, nx - x0,
                                                                                   src.at<Vec4b>(y0, x0)[b], src.at<Vec4b>(y0, x1)[b],
                                                                                   src.at<Vec4b>(y1, x0)[b], src.at<Vec4b>(y1, x1)[b]);
                        }
                            break;
                        default:
                            assert(false);
                            break;
                    }
                }
        }
    return dst;
}

void ImageWarp::setMappingPoints(const std::vector<cv::Point2f>& dst_points, const std::vector<cv::Point2f>& src_points)
{
    assert(src_points.size() == dst_points.size());

    this->src_points = dst_points;
    this->dst_points = src_points;
}

ImageWarp_Rigid::ImageWarp_Rigid()
{
    prescale = false;
}

static float calculateArea(const std::vector<cv::Point2f>& points)
{
    Vec4f box = boundingBox(points);
//	Vec4f(left, top, right, bottom);
    return (box[2] - box[0])*(box[3] - box[1]);
}

void ImageWarp_Rigid::calculateDelta(float alpha)
{
    const int N = static_cast<int>(src_points.size());

    float ratio;
    if(prescale)
    {
        // TODO use cv::contourArea(), the area is computed using the Green formula.
        float src_area = calculateArea(src_points);
        float dst_area = calculateArea(dst_points);
        ratio = std::sqrt(dst_area / src_area);

        for(int i = 0; i < N; ++i)
            dst_points[i] /= ratio;
    }

    std::vector<float> w(N);

    rDx.create(dst_size);
    rDy.create(dst_size);

    if(N < 2)
    {
        rDx.setTo(0);
        rDy.setTo(0);
        return;
    }

    for(int i = 0; ; i += grid_size)
    {
        if(i >= dst_size.width && i < dst_size.width + grid_size - 1)
            i = dst_size.width - 1;
        else if(i >= dst_size.width)
            break;

        for(int j = 0; ; j += grid_size)
        {
            if (j >= dst_size.height && j < dst_size.height + grid_size - 1)
                j = dst_size.height - 1;
            else if (j >= dst_size.height)
                break;

            float sw = 0;
            Point2f swp(0, 0), swq(0, 0), newP(0, 0), curV(i, j);

            int k;
            for(k = 0; k < N; ++k)
            {
                if((i == src_points[k].x) && j == src_points[k].y)
                    break;

//				float denorm = distance(Point2f(i, j), src_points[k]);
                float denorm = ((i - src_points[k].x)*(i - src_points[k].x) +
                                (j - src_points[k].y)*(j - src_points[k].y));
                if(alpha == 1.0F)
                    w[k] = 1.0F / denorm;
                else
                    w[k] = std::pow(denorm, -alpha);

                sw  = sw  + w[k];
                swp = swp + w[k] * src_points[k];
                swq = swq + w[k] * dst_points[k];
            }

            if(k == N)
            {
                Point2f pstar = swp / sw;
                Point2f qstar = swq / sw;

                // Calc miu_r
                // miu_s = 0;
                float s1 = 0, s2 = 0;
                for(k = 0; k < N; ++k)
                {
                    if (i == src_points[k].x && j == src_points[k].y)
                        continue;

                    Point2f Pi = src_points[k] - pstar;
                    Point2f PiJ;
                    PiJ.x = -Pi.y, PiJ.y = Pi.x;
                    Point2f Qi = dst_points[k] - qstar;
                    s1 += w[k] * Qi.dot(Pi);
                    s2 += w[k] * Qi.dot(PiJ);
                }

                float miu_r = std::sqrt(s1*s1 + s2*s2);
                curV -= pstar;
                Point2f curVJ;
                curVJ.x = -curV.y, curVJ.y = curV.x;

                for(k = 0; k < N; ++k)
                {
                    if(i==src_points[k].x && j==src_points[k].y)
                        continue;

                    Point2f Pi = src_points[k] - pstar;
                    Point2f PiJ;
                    PiJ.x = -Pi.y, PiJ.y = Pi.x;

                    Point2f tmpP;
                    tmpP.x = Pi.dot(curV) * dst_points[k].x - PiJ.dot(curV) * dst_points[k].y;
                    tmpP.y = -Pi.dot(curVJ) * dst_points[k].x + PiJ.dot(curVJ) * dst_points[k].y;
                    tmpP *= w[k]/miu_r;
                    newP += tmpP;
                }
                newP += qstar;
            }
            else
            {
                newP = dst_points[k];
            }

            if(prescale)
            {
                rDx(j, i) = newP.x * ratio - i;
                rDy(j, i) = newP.y * ratio - j;
            }
            else
            {
                rDx(j, i) = newP.x - i;
                rDy(j, i) = newP.y - j;
            }
        }
    }

    if(prescale)
    {
        for(int i = 0; i < N; ++i)
            dst_points[i] *= ratio;
    }
}



//眼影部分
void eyeshadow(cv::Mat src,std::vector<cv::Point> shape_points,cv::Mat tmp_left,cv::Mat tmp_right)
{
    //提取眼影关键点
    std::vector<cv::Point> eyepoint(12);
    if (!shape_points.empty())//左眼和右眼坐标
    {
        for (int i = 36; i <= 47; i++)
        {
            eyepoint[i-36].x=shape_points[i].x;
            eyepoint[i-36].y = shape_points[i].y;
        }
    }
    //模板输入
    //cv::Mat tmp_left = cv::imread("/storage/emulated/0/zuo3.png",cv::IMREAD_UNCHANGED);
    //cv::Mat tmp_right = cv::imread("/storage/emulated/0/you3.png", CV_LOAD_IMAGE_UNCHANGED);
    //int a[12][2] = { {132,117},{164,89},{204,96},{228,122},{205,127},{162,134},{50,116},{80,90},{116,85},{149,104},{120,123},{82,122} };
    int a[12][2] = { {132,114},{164,89},{203,95},{234,122},{205,130},{162,132},{50,117},{80,90},{116,88},{148,105},{120,123},{82,122} };
    std::vector<cv::Point> temppoint(12);
    for (int i = 0; i < 12; i++)//模板赋值
    {
        temppoint[i].x = a[i][0];
        temppoint[i].y = a[i][1];
    }
    //模板加长，不改变坐标轴，向下加长
    //cv::copyMakeBorder(tmp_left, tmp_left, 0, 50, 0, 0, cv::BORDER_CONSTANT, cv::Scalar( 255,255,255 ));
    //cv::copyMakeBorder(tmp_right, tmp_right, 0, 50, 0, 0, cv::BORDER_CONSTANT, cv::Scalar( 255,255,255));
    //cout << eyepoint[0].x<< " "<<eyepoint[0].y<<endl;
    //cout << eyepoint[3].x <<" "<< eyepoint[3].y << endl;

    //根据眼角距确定模板大小
    float scale_left = sqrt(pow((eyepoint[0].x - eyepoint[3].x), 2) + pow((eyepoint[0].y - eyepoint[3].y), 2)) / \
		            sqrt(pow((temppoint[0].x - temppoint[3].x), 2) + pow((temppoint[0].y - temppoint[3].y), 2));
    float scale_right = sqrt(pow((eyepoint[6].x - eyepoint[9].x), 2) + pow((eyepoint[6].y - eyepoint[9].y), 2)) / \
		sqrt(pow((temppoint[6].x - temppoint[9].x), 2) + pow((temppoint[6].y - temppoint[9].y), 2));
    cv::resize(tmp_left, tmp_left, Size(int(tmp_left.cols*scale_left), int(tmp_left.rows*scale_left)),0.0,0.0, CV_INTER_AREA);
    cv::resize(tmp_right, tmp_right, Size(int(tmp_right.cols*scale_right), int(tmp_right.rows*scale_right)), 0.0, 0.0, CV_INTER_AREA);
    //根据眼角距调节模板关键点坐标
    int len = int(temppoint.size());
    if (len % 2 != 0)
        return ;

    std::vector<cv::Point2d> p(len/2);
    std::vector<cv::Point2d> q(len / 2);
    for (int i = 0; i < len / 2; i++)//左眼
    {
        p[i].x = int(temppoint[i].x*scale_left);
        p[i].y = int(temppoint[i].y*scale_left);
    }

    std::vector<cv::Point2d> p_right(len / 2);
    std::vector<cv::Point2d> q_right(len / 2);
    for (int i = len / 2; i < len; i++)//右眼
    {
        p_right[i-len/2].x = int(temppoint[i].x*scale_right);
        p_right[i-len/2].y = int(temppoint[i].y*scale_right);
    }
    //旋转代替变形******************************************************************
    //以眼角为中心，旋转变化(缩小之后旋转),左眼
    float angle =float(temppoint[0].y - temppoint[3].y) / float(temppoint[0].x - temppoint[3].x) - float(eyepoint[0].y - eyepoint[3].y) / float(eyepoint[0].x - eyepoint[3].x);
    angle = atan(angle)*180/3.1415926;
    cv::Mat M = cv::getRotationMatrix2D(p[3], angle, 1);
    cv::warpAffine(tmp_left, tmp_left, M, cv::Size(tmp_left.cols, tmp_left.rows), cv::INTER_LINEAR,0,cv::Scalar(255,255,255));
    cv::transform(p, p, M);
    /*
    int x = 0, y = 0;
    for (int i = 0; i < len / 2; i++)//左眼
    {
        x = p[i].x - p[3].x;
        y = p[i].y - p[3].y;
        angle = angle *3.1415926/180;
        p[i].x = int(x * cos(angle) + y * sin(angle) + p[3].x);
        p[i].y =int( -x * sin(angle) + y * cos(angle) + p[3].y);
    }*/


    //右眼
    angle = float(temppoint[9].y - temppoint[6].y) / float(temppoint[9].x - temppoint[6].x) - float(eyepoint[9].y - eyepoint[6].y) / float(eyepoint[9].x - eyepoint[6].x);
    angle = atan(angle) * 180 / 3.1415926;
    M = cv::getRotationMatrix2D(p_right[0], angle, 1);
    cv::warpAffine(tmp_right, tmp_right, M, cv::Size(tmp_right.cols, tmp_right.rows), cv::INTER_LINEAR, 0, cv::Scalar(255, 255, 255));
    cv::transform(p_right, p_right, M);
    /*
    for (int i = 0; i < len / 2; i++)//右眼
    {
        x = p_right[i].x - p_right[0].x;
        y = p_right[i].y - p_right[0].y;
        angle = angle * 3.1415926 / 180;
        p_right[i].x = int(x * cos(angle) + y * sin(angle) + p_right[0].x);
        p_right[i].y = int(-x * sin(angle) + y * cos(angle) + p_right[0].y);
    }*/
    //****************************************


    //MLS眼边变化
    //左眼眼边,求偏移和q
    int shift1_left, shift2_left;

    shift1_left = eyepoint[3].x - p[3].x;
    shift2_left = eyepoint[3].y - p[3].y;
    for (int i = 0; i < len / 2; i++)
    {
        q[i].x = eyepoint[i].x - shift1_left;
        q[i].y = eyepoint[i].y - shift2_left;
    }
    //右眼眼边，求偏移和q_right
    int shift1_right, shift2_right;
    shift1_right = eyepoint[len/2].x - p_right[0].x;
    shift2_right = eyepoint[len/2].y - p_right[0].y;
    for (int i = 0; i < len/2 ; i++)
    {
        q_right[i].x = eyepoint[i+len/2].x - shift1_right;
        q_right[i].y = eyepoint[i+len/2].y - shift2_right;
    }
    //多点变形集成为单点
    std::vector<cv::Point2f> p1(2);
    std::vector<cv::Point2f> q1(2);

    q1[0].x = (q[1].x+q[2].x)/2;
    q1[0].y = (q[1].y+q[2].y)/2;
    q1[1].x = (q[4].x + q[5].x) / 2;
    q1[1].y = (q[4].y + q[5].y) / 2;
    p1[0].x = (p[1].x+p[2].x)/2;
    p1[0].y = (p[1].y+p[2].y)/2;
    p1[1].x = (p[4].x + p[5].x) / 2;
    p1[1].y = (p[4].y + p[5].y) / 2;

    std::vector<cv::Point2f> p_right1(2);
    std::vector<cv::Point2f> q_right1(2);

    q_right1[0].x = (q_right[1].x + q_right[2].x) / 2;
    q_right1[0].y = (q_right[1].y + q_right[2].y) / 2;
    q_right1[1].x = (q_right[4].x + q_right[5].x) / 2;
    q_right1[1].y = (q_right[4].y + q_right[5].y) / 2;
    p_right1[0].x = (p_right[1].x + p_right[2].x) / 2;
    p_right1[0].y = (p_right[1].y + p_right[2].y) / 2;
    p_right1[1].x = (p_right[4].x + p_right[5].x) / 2;
    p_right1[1].y = (p_right[4].y + p_right[5].y) / 2;
/*
    //调用变形算法1,MLS
    MLSD *deformobject;
    deformobject = new MLSDRigid();//刚性变化
    Mat dst_image = deformobject->Initialize(tmp_left, p1, q1,
                                             tmp_left.cols, tmp_left.rows);//左眼影

    Mat dst_image_right = deformobject->Initialize(tmp_right, p_right1, q_right1,
                                                   tmp_right.cols, tmp_right.rows);//右眼影
*/
//调用变形算法二
    ImageWarp_Rigid warp;
    warp.setMappingPoints(p1, q1);
    warp.setSourceSize(tmp_left.cols, tmp_left.rows);
    warp.setTargetSize(tmp_left.cols, tmp_left.rows);
    warp.calculateDelta(1.0F);
    Mat dst_image = warp.genNewImage(tmp_left, 1.0F);

    warp.setMappingPoints(p_right1, q_right1);
    warp.setSourceSize(tmp_right.cols, tmp_right.rows);
    warp.setTargetSize(tmp_right.cols, tmp_right.rows);
    warp.calculateDelta(1.0F);
    Mat dst_image_right = warp.genNewImage(tmp_right, 1.0F);
//变形结束

     /*融合方案一
    //通道变换，Lab空间融合
    cv::cvtColor(src, src, CV_BGR2Lab);
    cv::cvtColor(dst_image, dst_image, CV_BGR2Lab);
    cv::cvtColor(dst_image_right, dst_image_right, CV_BGR2Lab);

    //遍历目标像素点融合
    for (int row = 0; row < dst_image.rows; row++)//左
    {
        for (int col = 0; col < dst_image.cols; col++)
        {
            //if (i == 0 && j == 0)
            //cout <<(int)dst_image.at<Vec3b>(row,col)[0]<<" ";
            //cout <<src.at<Vec4b>(i, j)[1];
            if (dst_image.at<Vec3b>(row,col)[0] < 227&& dst_image.at<Vec3b>(row, col)[0]>50)
            {
                src.at<Vec3b>(row+ shift2_left, col + shift1_left)[1] = dst_image.at<Vec3b>(row, col)[1] + 14;
                src.at<Vec3b>(row + shift2_left, col + shift1_left)[2] = dst_image.at<Vec3b>(row, col)[2] + 11;
            }
        }
    }
    for (int row = 0; row < dst_image_right.rows; row++)//右
    {
        for (int col = 0; col < dst_image_right.cols; col++)
        {
            if (dst_image_right.at<Vec3b>(row, col)[0] < 227 && dst_image_right.at<Vec3b>(row, col)[0]>50)
            {
                src.at<Vec3b>(row + shift2_right, col + shift1_right)[1] = dst_image_right.at<Vec3b>(row, col)[1] + 14;
                src.at<Vec3b>(row + shift2_right, col + shift1_right)[2] = dst_image_right.at<Vec3b>(row, col)[2] + 11;
            }
        }
    }
    cv::cvtColor(src, src, CV_Lab2BGR);
   */
     //融合方案二，渐进效果
    int color[3] = { 184,44,149 };//RGB
    //左
    int height = dst_image.rows;
    int width = dst_image.cols;
    int step1 = dst_image.step;
    int channels = dst_image.channels();
    float tmp = 0;//调节色彩的浓淡，即融合比例
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {

            tmp = 0.11*dst_image.data[i*step1 + channels * j] + 0.59*dst_image.data[i*step1 + channels * j + 1] + 0.3*dst_image.data[i*step1 + channels * j + 2];
            tmp = 0.8*(1-tmp / 255.0);
            if (tmp > 1)
                tmp = 1;
            //cout << tmp << ' ';
            if ((i + shift2_left) <= 0 || (j + shift1_left) <= 0)
                continue;
            src.at<Vec3b>(i + shift2_left, j + shift1_left)[0] =int (src.at<Vec3b>(i + shift2_left, j + shift1_left)[0]*(1-tmp)+/*dst_image.at<Vec3b>(i, j)[0]*/color[2]*tmp);
            src.at<Vec3b>(i + shift2_left, j + shift1_left)[1] = int(src.at<Vec3b>(i + shift2_left, j + shift1_left)[1] * (1 - tmp) + /*dst_image.at<Vec3b>(i, j)[1] */color[1]* tmp);
            src.at<Vec3b>(i + shift2_left, j + shift1_left)[2] = int(src.at<Vec3b>(i + shift2_left, j + shift1_left)[2] * (1 - tmp) + /*dst_image.at<Vec3b>(i, j)[2] */color[0]* tmp);
        }

    //右
    height = dst_image_right.rows;
    width = dst_image_right.cols;
    step1 = dst_image_right.step;
    channels = dst_image_right.channels();
    tmp = 0;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {

            tmp = 0.11*dst_image_right.data[i*step1 + channels * j] + 0.59*dst_image_right.data[i*step1 + channels * j + 1] + 0.3*dst_image_right.data[i*step1 + channels * j + 2];
            tmp = 0.8*(1 - tmp / 255.0);
            if (tmp > 1)
                tmp = 1;
            if ((i + shift2_right) <= 0 || (j + shift1_right) <= 0)
                continue;
            src.at<Vec3b>(i + shift2_right, j + shift1_right)[0] = int(src.at<Vec3b>(i + shift2_right, j + shift1_right)[0] * (1 - tmp) +color[2] * tmp);
            src.at<Vec3b>(i + shift2_right, j + shift1_right)[1] = int(src.at<Vec3b>(i + shift2_right, j + shift1_right)[1] * (1 - tmp) +color[1] * tmp);
            src.at<Vec3b>(i + shift2_right, j + shift1_right)[2] = int(src.at<Vec3b>(i + shift2_right, j + shift1_right)[2] * (1 - tmp) + color[0] * tmp);
        }
}