#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

namespace image_processing {
    // detection
    void hough_circle_detection(cv::Mat input_image, bool debug=false);
    cv::Mat sobel_edge(cv::Mat input_img);
    // image mutations and operations
    cv::Mat pad_image(cv::Mat input_img, uint32_t pad_height, uint32_t pad_width);
    cv::Mat conv_image_2d_3chan(cv::Mat input_img, vector<vector<int>> kernel, int kernel_size, bool padding=false);
    cv::Mat conv_image_2d_1chan(cv::Mat input_img, vector<vector<int>> kernel, int kernel_size, bool padding=false);
    cv::Mat conv_laplacian(cv::Mat input_img);
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
}

#endif // IMAGE_PROCESSING_H
