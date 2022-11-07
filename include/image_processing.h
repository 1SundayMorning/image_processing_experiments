#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

namespace image_processing {
    // detection
    void hough_circle_detection(cv::Mat input_img);
    // image mutations and operations
    cv::Mat pad_image(cv::Mat input_img, uint32_t pad_height, uint32_t pad_width);
    cv::Mat conv_image_2d(cv::Mat input_img, vector<vector<int32_t>> kernel);
    // image filters
    cv::Mat integrated_gaussian_blur(cv::Mat input_img, double sigma, uint32_t kernel_size);
    cv::Mat gaussian_blur(cv::Mat input_img, double** kernel, uint32_t kernel_size);
    // image queries
    unsigned int get_average_pixel_intensity(cv::Mat grey_img);
    // thresholding functions
    cv::Mat color_img_to_thresh(cv::Mat input_img);
    cv::Mat color_img_to_thresh(cv::Mat input_img, uint32_t setpoint);
    //utility
    void create_gaussian_kernel(double sigma, double** kernel, uint32_t kernel_size);
}

#endif // IMAGE_PROCESSING_H
