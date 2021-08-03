#include <iostream>
#include <string>
#include <cstdlib>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include "cv.h"
#include "highgui.h"
#include "math.h"
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <time.h>
#include <iostream>
#include <set>

using namespace std;
using namespace cv;

RNG rng(12345);
float getDistance(CvPoint pointO, CvPoint pointA);
float getAngle(CvPoint pointM, CvPoint pointL, CvPoint pointR);

float getDist_P2L(CvPoint pointP, CvPoint pointA, CvPoint pointB);
int list_connor(int i1, int i2, int i3);
bool flag = true;

cv::Mat HSV;
cv::Mat threshold_ori;

int Otsu(IplImage *src)
{
    int height = src->height;
    int width = src->width;

    //histogram
    float histogram[256] = {0};
    for (int i = 0; i < height; i++)
    {
        unsigned char *p = (unsigned char *)src->imageData + src->widthStep * i;
        for (int j = 0; j < width; j++)
        {
            histogram[*p++]++;
        }
    }
    //normalize histogram
    int size = height * width;
    for (int i = 0; i < 256; i++)
    {
        histogram[i] = histogram[i] / size;
    }

    //average pixel value
    float avgValue = 0;
    for (int i = 0; i < 256; i++)
    {
        avgValue += i * histogram[i]; //整幅图像的平均灰度
    }

    int threshold;
    float maxVariance = 0;
    float w = 0, u = 0;
    for (int i = 0; i < 256; i++)
    {
        w += histogram[i];     //假设当前灰度i为阈值, 0~i 灰度的像素(假设像素值在此范围的像素叫做前景像素) 所占整幅图像的比例
        u += i * histogram[i]; // 灰度i 之前的像素(0~i)的平均灰度值： 前景像素的平均灰度值

        float t = avgValue * w - u;
        float variance = t * t / (w * (1 - w));
        if (variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }

    return threshold;
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    //获取图像
    cv::Mat cameraFeed;
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); // Caution the type here.
    }
    catch (cv_bridge::Exception &ex)
    {
        ROS_ERROR("cv_bridge exception in rgbcallback: %s", ex.what());
        exit(-1);
    }
    cameraFeed = cv_ptr->image.clone();

    //转为黑白
    IplImage temp = IplImage(cameraFeed);
    IplImage *ImgGray = &temp;
    IplImage *img = cvCreateImage(cvGetSize(ImgGray), IPL_DEPTH_8U, 1);
    cvCvtColor(ImgGray, img, CV_BGR2GRAY); //cvCvtColor(src,des,CV_BGR2GRAY)
    IplImage *dst = cvCreateImage(cvGetSize(img), 8, 1);
    int threshold = Otsu(img);
    cvThreshold(img, dst, threshold, 255, CV_THRESH_BINARY);

    //显示原图
    cv::Mat srcImage;
    srcImage = cvarrToMat(dst);
    imshow("原图", srcImage);

    //getStructuringElement函数会返回指定形状和尺寸的结构元素
    //矩形：MORPH_RECT;
    //交叉形：MORPH_CROSS;
    //椭圆形：MORPH_ELLIPSE;
    // Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    //morphologyEx(srcImage, srcImage, MORPH_CLOSE, element);//闭运算滤波
    vector<vector<Point>> contours, RectContours;                         //轮廓，为点向量，
    findContours(srcImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE); //找轮廓
    std::cout << "contours.size()" << contours.size() << std::endl;

    vector<vector<Point>> hull(contours.size()); //用于存放凸包
    Mat drawing(srcImage.size(), CV_8UC3, cv::Scalar(0));
    int i = 0;
    vector<float> length(contours.size()); //用于保存每个轮廓的长度
    vector<float> Area_contours(contours.size()), Area_hull(contours.size()), Rectangularity(contours.size()), circularity(contours.size());
    for (i = 0; i < contours.size(); i++)
    { //把所有的轮廓画出来
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        length[i] = arcLength(contours[i], true); //轮廓的长度
        //通过长度匹配滤除小轮廓
        if (length[i] > 200 && length[i] < 2000)
        {
            convexHull(Mat(contours[i]), hull[i], false);                               //把凸包找出来，寻找凸包函数
            Area_contours[i] = contourArea(contours[i]);                                //轮廓面积
            Area_hull[i] = contourArea(hull[i]);                                        //凸包面积
            Rectangularity[i] = Area_contours[i] / Area_hull[i];                        //矩形度
            circularity[i] = (4 * 3.1415 * Area_contours[i]) / (length[i] * length[i]); //圆形度
            //drawContours(drawing, contours, i, color, 1);//得到方框

            if (Rectangularity[i] > 0.8 && circularity[i] < 0.9)
            { //通过矩形度和圆形度滤除数字
                //drawContours(drawing, contours, i, Scalar(255, 255, 255), 1);
                RectContours.push_back(hull[i]);          //把提取出来的方框导入到新的轮廓组
                drawContours(drawing, hull, i, color, 1); //得到方框
            }
        }
    }

    float distance = 0, distanceMax = 0;
    Point connorPoint1, connorPoint2, connorPoint3, connorPoint4, point_add;
    vector<Point> connor4_add(3); //先找到的三个角点
    vector<Point> corner(4);      //存储点
    int conP_i1, conP_i2, conP_i3, conP_i_add;
    int j = 0, flag = 0;

    Point finally_contours[80][4];            //轮廓，为点向量，新的轮廓
    for (j = 0; j < RectContours.size(); j++) //四边形轮廓个数
    {
        distance = 0;
        distanceMax = 0;
        for (i = 0; i < RectContours[j].size(); i++) //每个轮廓点的个数11到19点不等
        {                                            //找第一个角点
            distance = getDistance(RectContours[j][i], RectContours[j][0]);
            if (distance > distanceMax)
            {
                distanceMax = distance;
                connorPoint1 = RectContours[j][i]; //第一个角点
                conP_i1 = i;
            }
        }
        distance = 0;
        distanceMax = 0;
        for (i = 0; i < RectContours[j].size(); i++)
        { //找第二个角点
            distance = getDistance(RectContours[j][i], connorPoint1);
            if (distance > distanceMax)
            {
                distanceMax = distance;
                connorPoint2 = RectContours[j][i]; //第二个角点
                conP_i2 = i;
            }
        }
        distance = 0;
        distanceMax = 0;
        for (i = 0; i < RectContours[j].size(); i++)
        { //找第三个角点
            distance = getDistance(RectContours[j][i], connorPoint1) + getDistance(RectContours[j][i], connorPoint2);
            if (distance > distanceMax)
            {
                distanceMax = distance;
                connorPoint3 = RectContours[j][i]; //第三个角点
                conP_i3 = i;
            }
        }
        flag = list_connor(conP_i1, conP_i2, conP_i3); //对三个角点由大到小排序
        switch (flag)
        { //对三个角点排序
        case 0:
            break;
        case 123:
            break;
        case 132:
            point_add = connorPoint2;
            connorPoint2 = connorPoint3;
            connorPoint3 = point_add;
            break; //2,3交换
        case 213:
            point_add = connorPoint1;
            connorPoint1 = connorPoint2;
            connorPoint2 = point_add;
            break; //1,2交换
        case 231:
            point_add = connorPoint1;
            connorPoint1 = connorPoint2;
            connorPoint2 = point_add;
            point_add = connorPoint2;
            connorPoint2 = connorPoint3;
            connorPoint3 = point_add;
            break; //1,2交换+2,3交换
        case 321:
            point_add = connorPoint3;
            connorPoint3 = connorPoint1;
            connorPoint1 = point_add;
            break; //1,3交换
        case 312:
            point_add = connorPoint3;
            connorPoint3 = connorPoint1;
            connorPoint1 = point_add;
            point_add = connorPoint2;
            connorPoint2 = connorPoint3;
            connorPoint3 = point_add;
            break; //1,3交换+2,3交换
        }
        switch (flag)
        { //对三个角点排序
        case 0:
            break;
        case 123:
            break;
        case 132:
            conP_i_add = conP_i2;
            conP_i2 = conP_i3;
            conP_i3 = conP_i_add;
            break; //2,3交换
        case 213:
            conP_i_add = conP_i1;
            conP_i1 = conP_i2;
            conP_i2 = conP_i_add;
            break; //1,2交换
        case 231:
            conP_i_add = conP_i1;
            conP_i1 = conP_i2;
            conP_i2 = conP_i_add;
            conP_i_add = conP_i2;
            conP_i2 = conP_i3;
            conP_i3 = conP_i_add;
            break; //1,2交换+2,3交换
        case 321:
            conP_i_add = conP_i3;
            conP_i3 = conP_i1;
            conP_i1 = conP_i_add;
            break; //1,3交换
        case 312:
            conP_i_add = conP_i3;
            conP_i3 = conP_i1;
            conP_i1 = conP_i_add;
            conP_i_add = conP_i2;
            conP_i2 = conP_i3;
            conP_i3 = conP_i_add;
            break; //1,3交换+2,3交换
        }
        distance = 0;
        distanceMax = 0;
        for (i = conP_i3; i < conP_i2; i++)
        { //相隔两角点之间找到怀疑是4角点的点
            distance = getDistance(RectContours[j][i], connorPoint3) + getDistance(RectContours[j][i], connorPoint2);
            if (distance > distanceMax)
            {
                distanceMax = distance;
                connor4_add[0] = RectContours[j][i];
            }
        }
        distance = 0;
        distanceMax = 0;
        for (i = conP_i2; i < conP_i1; i++)
        { //相隔两角点之间找到怀疑是4角点的点
            distance = getDistance(RectContours[j][i], connorPoint1) + getDistance(RectContours[j][i], connorPoint2);
            if (distance > distanceMax)
            {
                distanceMax = distance;
                connor4_add[1] = RectContours[j][i];
            }
        }
        distance = 0;
        distanceMax = 0;
        for (i = conP_i1; i < RectContours[j].size() + conP_i3; i++)
        { //相隔两角点之间找到怀疑是4角点的点
            if (i < RectContours[j].size())
            {
                distance = getDistance(RectContours[j][i], connorPoint1) + getDistance(RectContours[j][i], connorPoint3);
                if (distance > distanceMax)
                {
                    distanceMax = distance;
                    connor4_add[2] = RectContours[j][i];
                }
            }
            else
            {
                distance = getDistance(RectContours[j][i - RectContours[j].size()], connorPoint1) + getDistance(RectContours[j][i - RectContours[j].size()], connorPoint3);
                if (distance > distanceMax)
                {
                    distanceMax = distance;
                    connor4_add[2] = RectContours[j][i - RectContours[j].size()];
                }
            }
        }

        if (getDist_P2L(connor4_add[0], connorPoint3, connorPoint2) > 10)
        {
            connorPoint4 = connor4_add[0];
        }
        else if (getDist_P2L(connor4_add[1], connorPoint2, connorPoint1) > 10)
        {
            connorPoint4 = connor4_add[1];
        }
        else if (getDist_P2L(connor4_add[2], connorPoint1, connorPoint3) > 10)
        {
            connorPoint4 = connor4_add[2];
        }
        corner.clear();
        corner[0]=connorPoint1;
        corner[1]=connorPoint2;
        corner[2]=connorPoint3;
        corner[3]=connorPoint4;
        float theta = 0;
        float max_theta = 0;
        // std::cout << "theta................................" << std::endl;
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<4;j++)
            {
                for (int z = 0; z < 4; z++)
                {
                    if(i!=j && i!=z &&z!=j)
                    {
                        // std::cout << "ijz:" << i<<j<<z << std::endl;
                        theta = getAngle(corner[i], corner[j], corner[z]);
                        theta = theta / 3.1415 * 180;
                        if(theta>max_theta)
                        {
                            max_theta = theta;
                        }
                        // std::cout << "theta:" << theta << std::endl;
                    }
                }
            }
        }
        if(max_theta>130)
        {
            continue;
        }

        circle(drawing, connorPoint1, 3, Scalar(255, 255, 255), FILLED, LINE_AA);
        circle(drawing, connorPoint2, 3, Scalar(255, 255, 255), FILLED, LINE_AA);
        circle(drawing, connorPoint3, 3, Scalar(255, 255, 255), FILLED, LINE_AA);
        circle(drawing, connorPoint4, 3, Scalar(255, 255, 255), FILLED, LINE_AA);

        finally_contours[j][0] = connorPoint1;
        finally_contours[j][1] = connorPoint2;
        finally_contours[j][2] = connorPoint3;
        finally_contours[j][3] = connorPoint4;

        cout << "\n轮廓 " << j + 1 << " 的四个角点坐标分别为：\n"
             << finally_contours[j][0] << finally_contours[j][1] << finally_contours[j][2] << finally_contours[j][3] << endl;
    }

    imshow("轮廓", drawing);
    cvWaitKey(10000);
}

float getDist_P2L(CvPoint pointP, CvPoint pointA, CvPoint pointB)
{
    //点到直线的距离:P到AB的距离
    //求直线方程
    int A = 0, B = 0, C = 0;
    A = pointA.y - pointB.y;
    B = pointB.x - pointA.x;
    C = pointA.x * pointB.y - pointA.y * pointB.x;
    //代入点到直线距离公式
    float distance = 0;
    distance = ((float)abs(A * pointP.x + B * pointP.y + C)) / ((float)sqrtf(A * A + B * B));
    return distance;
}

//对角点进行排序，因为之前检测出的轮廓是带序号的
int list_connor(int i1, int i2, int i3)
{
    //排序
    int flag = 0;
    Point point_add;
    if (i1 >= i2 && i2 >= i3)
        flag = 123;
    else if (i1 >= i3 && i3 >= i2)
        flag = 132;
    else if (i2 >= i1 && i1 >= i3)
        flag = 213;
    else if (i2 >= i3 && i3 >= i1)
        flag = 231;
    else if (i3 >= i2 && i2 >= i1)
        flag = 321;
    else if (i3 >= i1 && i1 >= i2)
        flag = 312;
    return flag;
}

float getDistance(CvPoint pointO, CvPoint pointA)
{
    //求两点之间距离
    float distance;
    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
    distance = sqrtf(distance);
    return distance;
}
float getAngle(CvPoint pointM, CvPoint pointL, CvPoint pointR)
{
    //求三点之间的夹角
    CvPoint L, R;
    float dist_L, dist_R, Theta;
    L.x = pointL.x - pointM.x;
    L.y = pointL.y - pointM.y;
    R.x = pointR.x - pointM.x;
    R.y = pointR.y - pointM.y;
    dist_L = getDistance(pointL, pointM);
    dist_R = getDistance(pointR, pointM);
    Theta = acos((L.x * R.x + L.y * R.y) / (dist_L * dist_R));
    return Theta;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_camera_extrinsic_calibration_node");
    ros::NodeHandle node;
    ros::NodeHandle n("~");

    ros::Subscriber sub = n.subscribe("/camera/image_color", 1000, imageCallback);
    ros::spin();

    return 0;
}
