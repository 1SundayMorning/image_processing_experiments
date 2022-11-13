#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "image_processing.h"
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {

    string path = "images/coins.jpeg";
    Mat raw_img = imread(path);

    vector<vector<vector<int>>> circles;
    circles = image_processing::hough_circle_detection(raw_img, 8.1, 15, 20, 60);

    Mat result_circles = image_processing::draw_circles(raw_img, circles);

    imshow("result circles", result_circles);

    // Mat circles = image_processing::opencv_hough(raw_img);

    // imshow("circles image", circles);
    // // gauss blur image
    // Mat gauss_img = image_processing::gaussian_blur(raw_img, 3, 7);
    // // take image thresh
    // Mat thresh_img = image_processing::convert_binary_image(raw_img);

    // imshow("gauss image", gauss_img);
    // imshow("thresh image", thresh_img);
    waitKey(0);
    return 0;
}
