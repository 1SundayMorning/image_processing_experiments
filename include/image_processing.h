#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>

using namespace std;

namespace image_processing {
    // detection
    vector<vector<vector<int>>> hough_circle_detection(cv::Mat input_image, float threshold, int region, int min_radius, int max_radius, bool debug=false);
    cv::Mat sobel_edge(cv::Mat input_img);
    cv::Mat laplacian_edge(cv::Mat input_img);
    cv::Mat opencv_hough(cv::Mat input_img);
    // image mutations and operations
    cv::Mat pad_image(cv::Mat input_img, uint32_t pad_height, uint32_t pad_width);
    cv::Mat conv_image_2d_3chan(cv::Mat input_img, vector<vector<int>> kernel, int kernel_size, bool padding=false);
    cv::Mat conv_image_2d_1chan(cv::Mat input_img, vector<vector<int>> kernel, int kernel_size, bool padding=false);
    cv::Mat conv_gaussian(cv::Mat input_img, double** kernel, uint32_t kernel_size);
    // image filters
    cv::Mat gaussian_blur(cv::Mat input_img, double sigma, uint32_t kernel_size);
    // image queries
    unsigned int get_average_pixel_intensity(cv::Mat grey_img);
    // thresholding functions
    cv::Mat convert_binary_image(cv::Mat input_img);
    cv::Mat convert_binary_image(cv::Mat input_img, uint32_t setpoint);
    //utility
    void create_gaussian_kernel(double sigma, double** kernel, uint32_t kernel_size);
    cv::Mat draw_circles(cv::Mat input_image, vector<vector<vector<int>>> input_circles);
}

#endif // IMAGE_PROCESSING_H
