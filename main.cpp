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

    image_processing::hough_circle_detection(raw_img);
    
    // // gauss blur image
    // Mat gauss_img = image_processing::gaussian_blur(raw_img, 3, 7);
    // // take image thresh
    // Mat thresh_img = image_processing::convert_binary_image(raw_img);

    // imshow("gauss image", gauss_img);
    // imshow("thresh image", thresh_img);
    waitKey(0);
    return 0;
}
