#include <jni.h>
#include <string>
#include <iostream>
#include <dlib/matrix.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "dlib_proc.h"
#include "skin_makeup.h"
#include "lip_makeup.h"
#include "eyeshadow.h"
#include "eyebrow.h"
#include "eyeliner.h"
#include "eyelash.h"
#include "blush.h"
#include <android/log.h>
#include "jni_bridge.h"

using namespace dlib;
using namespace std;
using namespace cv ;
extern "C" JNIEXPORT jintArray


JNICALL
Java_com_example_ai_dlibdemo_MainActivity_Bitmap2proc(JNIEnv *env, jobject /* this */,
                                                      jintArray buf, jintArray eye_left,
                                                      jintArray eye_right,
                                                      jintArray p_eyebrow_left,
                                                      jintArray p_eyebrow_right,
                                                      jintArray blush_l,
                                                      jintArray blush_r,
                                                      jobject lash,
                                                      jintArray w, jintArray h,
                                                      jintArray points, jint type,
                                                      jint left, jint top, jint right, jint bootom) {
    //声明jc+指针
    jint *cbuf, *cw, *ch, *ceye_left, *ceye_right, *cleftb, *crightb, *cblush_l, *cblush_r;
    jboolean ptfalse = false;


    AndroidBitmapInfo mask_info;
    uint32_t *mask_pixels = lockJavaBitmap(env, lash, mask_info);
    assert(mask_pixels != nullptr && mask_info.format == ANDROID_BITMAP_FORMAT_A_8);
    Mat mask(mask_info.height, mask_info.width, CV_8UC1, mask_pixels);


    cbuf = env->GetIntArrayElements(buf, &ptfalse);
    cw = env->GetIntArrayElements(w, &ptfalse);
    ch = env->GetIntArrayElements(h, &ptfalse);
    ceye_left = env->GetIntArrayElements(eye_left, &ptfalse);
    ceye_right = env->GetIntArrayElements(eye_right, &ptfalse);
    cleftb = env->GetIntArrayElements(p_eyebrow_left, &ptfalse);
    crightb = env->GetIntArrayElements(p_eyebrow_right, &ptfalse);
    cblush_l = env->GetIntArrayElements(blush_l, &ptfalse);
    cblush_r = env->GetIntArrayElements(blush_r, &ptfalse);

    jsize len = env->GetArrayLength(points);
    if (cbuf == NULL || len <= 0) {
        return 0;
    }


    Mat imgData(ch[0], cw[0], CV_8UC4, (unsigned char *) cbuf);
    Mat ret_imgData(ch[0], cw[0], CV_8UC4);
    Mat leyeData(ch[1], cw[1], CV_8UC4, (unsigned char *) ceye_left);
    Mat reyeData(ch[2], cw[2], CV_8UC4, (unsigned char *) ceye_right);
    Mat lbrowData(ch[3], cw[3], CV_8UC4, (unsigned char *) cleftb);
    Mat rbrowData(ch[4], cw[4], CV_8UC4, (unsigned char *) crightb);
    Mat lblushData(ch[5], cw[5], CV_8UC4, (unsigned char *) cblush_l);
    Mat rblushData(ch[6], cw[6], CV_8UC4, (unsigned char *) cblush_r);

    //Mat half_imgData;
    // resize(imgData, half_imgData,Size(),0.3,0.3);
    cvtColor(imgData, imgData, CV_BGRA2BGR);
    cvtColor(leyeData, leyeData, CV_BGRA2BGR);
    cvtColor(reyeData, reyeData, CV_BGRA2BGR);
    cvtColor(lbrowData, lbrowData, CV_BGRA2BGR);
    cvtColor(rbrowData, rbrowData, CV_BGRA2BGR);

    jintArray array = env->NewIntArray(len);
    jint *body = env->GetIntArrayElements(points, 0);

    std::vector<Point> shape_points;
    std::vector<Point> face_points;
    for (int i = 0; i < len / 2; i++) {
        shape_points.push_back(Point(body[2 * i], body[2 * i + 1]));
        face_points.push_back(Point(body[2 * i] - left, body[2 * i + 1] - top));
    }
    Mat facedata = imgData(Rect(left, top, right - left, bootom - top));
    switch (type) {
        case 1:
            //嘴唇化妆
            makeLip(imgData, shape_points);

            //眼影
            eyeshadow(imgData, shape_points, leyeData, reyeData);

            //眉毛
            eyebrow(imgData, shape_points, lbrowData, rbrowData);

            //眼线
            eyeliner_makeup(imgData, shape_points);

            //睫毛
            cvtColor(imgData, imgData, CV_BGR2BGRA);
            eyelash(imgData, mask, shape_points);
            cvtColor(imgData, imgData, CV_BGRA2BGR);

            //腮红
            blush(imgData, shape_points, lblushData, rblushData);
            //磨皮
            smooth_skin(imgData);

            break;
        case 2:

            //嘴唇化妆
            makeLip(imgData, shape_points);
            break;
        case 3:
            //眼影
            eyeshadow(imgData, shape_points, leyeData, reyeData);
            break;
        case 4:
            //眉毛
            eyebrow(imgData, shape_points, lbrowData, rbrowData);
            break;
        case 5:
            //磨皮
            smooth_skin(imgData);
            break;

        case 6:
            //睫毛
            cvtColor(imgData, imgData, CV_BGR2BGRA);
            eyelash(imgData, mask, shape_points);
            cvtColor(imgData, imgData, CV_BGRA2BGR);
            break;

        case 7:
            //腮红
            blush(imgData, shape_points, lblushData, rblushData);
            break;

        case 8:
            //眼线
            eyeliner_makeup(imgData, shape_points);

            break;
        case 9:
            //瘦脸
            

            break;
        default:
            break;
    }
//    switch (type) {
//        case 1:
//            //嘴唇化妆
//            makeLip(facedata, face_points);
//
//            //眼影
//            eyeshadow(facedata, face_points, leyeData, reyeData);
//
//            //眉毛
//            eyebrow(facedata, face_points);
//
//            //磨皮
//            smooth_skin(facedata);
//            break;
//        case 2:
//
//            //嘴唇化妆
//            makeLip(facedata, face_points);
//            break;
//        case 3:
//            //眼影
//            eyeshadow(facedata, face_points, leyeData, reyeData);
//            break;
//        case 4:
//            //眉毛
//            eyebrow(facedata, face_points);
//            break;
//        case 5:
//            //磨皮
//            smooth_skin(facedata);
//            break;
//        default:
//            break;
//    }

    resize(imgData, imgData, Size(cw[0], ch[0]));
    cvtColor(imgData, ret_imgData, CV_BGR2BGRA);
    int size = cw[0] * ch[0];
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, (jint *) ret_imgData.data);
    env->ReleaseIntArrayElements(buf, cbuf, 0);
    unlockJavaBitmap(env, lash);
    return result;
}


//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ai_dlibdemo_TestActivcity_Makeup(
//        JNIEnv *env, jobject instance, jobject bitmap1, jobject bitmap2, jobject lash) {


//    AndroidBitmapInfo dst_info;
//    uint32_t *dst_pixels = lockJavaBitmap(env, bitmap1, dst_info);
//    Mat dst(dst_info.height, dst_info.width, CV_8UC4, dst_pixels);
//
//    AndroidBitmapInfo src_info;
//    uint32_t *src_pixels = lockJavaBitmap(env, bitmap2, src_info);
//    Mat src(src_info.height, src_info.width, CV_8UC4, src_pixels);
//
//    AndroidBitmapInfo mask_info;
//    uint32_t *mask_pixels = lockJavaBitmap(env, lash, mask_info);
//    assert(mask_pixels != nullptr && mask_info.format == ANDROID_BITMAP_FORMAT_A_8);
//    Mat mask(mask_info.height, mask_info.width, CV_8UC1, mask_pixels);
//
//    std::vector<Point> shape_points;
//    eyelash(dst, mask, shape_points);
//    // applyEyeLash(dst, src, points, mask, color, amount);
//
//    unlockJavaBitmap(env, bitmap1);
//    unlockJavaBitmap(env, bitmap2);
//    unlockJavaBitmap(env, lash);
//}
