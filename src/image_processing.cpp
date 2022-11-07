#include "image_processing.h"

#define PI 3.14159

void image_processing::hough_circle_detection(cv::Mat input_img) {
    // gaussian blur image
    double sigma = 3;
    uint32_t kernel_size = 7;
    cv::Mat guass_image = image_processing::integrated_gaussian_blur(input_img, sigma, kernel_size);
    cv::imshow("gauss_image", gauss_image);
    
}

cv::Mat image_processing::pad_image(cv::Mat input_img, uint32_t pad_height, uint32_t pad_width) {
    uint32_t image_height = input_img.rows;
    uint32_t image_width = input_img.cols;
    cv::Mat padded_img = cv::Mat(image_height + 2 * pad_height, image_width + 2 * pad_width, input_img.type(), 255);

    uint8_t channels = input_img.channels();

    for (int i = pad_height; i < pad_height + image_height; i++) {
        for (int j = pad_width; j < pad_width + image_width; j++) {
            if(channels == 3) {
                padded_img.at<cv::Vec3b>(i,j) = input_img.at<cv::Vec3b>(i - pad_height, j - pad_width);
            }
            else if (channels == 1) {
                padded_img.at<uchar>(i,j) = input_img.at<uchar>(i - pad_height, j - pad_width);
            }
        }
    }
    return padded_img;
}

cv::Mat image_processing::conv_image_2d(cv::Mat input_img, vector<vector<int32_t>> kernel) {
    return cv::Mat();
}

unsigned int image_processing::get_average_pixel_intensity(cv::Mat grey_img) {
    uint32_t total_pixels = grey_img.rows * grey_img.cols;
    uint32_t intensity_sum = 0;
    for (int i = 0; i < grey_img.rows; i++) {
        for (int j = 0; j < grey_img.cols; j++) {
            intensity_sum += (uint32_t)grey_img.at<uchar>(i,j);
        }
    }
    return intensity_sum/total_pixels;
}

cv::Mat image_processing::color_img_to_thresh(cv::Mat input_img) {
    cv::Mat grey_img;
    cv::cvtColor(input_img, grey_img, cv::COLOR_BGR2GRAY);

    uint32_t thresh_setpoint = get_average_pixel_intensity(grey_img);

    for (int i = 0; i < grey_img.rows; i++) {
        for (int j = 0; j < grey_img.cols; j++) {
            if ((uint32_t)grey_img.at<uchar>(i,j) < thresh_setpoint) {
                grey_img.at<uchar>(i,j) = 0;
            }
            else {
                grey_img.at<uchar>(i,j) = 255;
            }
        }
    }
    return grey_img;
}

cv::Mat image_processing::color_img_to_thresh(cv::Mat input_img, uint32_t setpoint) {
    cv::Mat grey_img;
    cv::cvtColor(input_img, grey_img, cv::COLOR_BGR2GRAY);

    for (int i = 0; i < grey_img.rows; i++) {
        for (int j = 0; j < grey_img.cols; j++) {
            if ((uint32_t)grey_img.at<uchar>(i,j) < setpoint) {
                grey_img.at<uchar>(i,j) = 0;
            }
            else {
                grey_img.at<uchar>(i,j) = 255;
            }
        }
    }
    return grey_img;
}

cv::Mat image_processing::integrated_gaussian_blur(cv::Mat input_img, double sigma, uint32_t kernel_size) {
    double ** gauss_kernel;
    gauss_kernel = new double*[kernel_size];
    for (int i = 0; i < kernel_size; i++) {
        gauss_kernel[i] = new double[kernel_size];
    }
    image_processing::create_gaussian_kernel(sigma, gauss_kernel, kernel_size);
    cv::Mat gauss_img = image_processing::gaussian_blur(input_img, gauss_kernel, kernel_size);

    //deallocate gaussian kernel
    for (int i = 0; i < kernel_size; i++) {
        free(gauss_kernel[i]);
    }
    free(gauss_kernel);

    return gauss_img;
}

void image_processing::create_gaussian_kernel(double sigma, double** kernel, uint32_t kernel_size) {
    double p = 2 * sigma * sigma;
    double q = p;
    double sum = 0;

    int neg_range = (kernel_size/2) * - 1;
    int pos_range = (kernel_size/2);

    for (int x = neg_range; x <= pos_range; x++) {
        for (int y = neg_range; y <= pos_range; y++) {
            p = sqrt(x * x + y * y);
            kernel[x + pos_range][y + pos_range] = (exp(-(p * p) / q)) / (PI * q);
            sum += kernel[x + pos_range][y + pos_range];
        }
    }

    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] /= sum;
        }
    }

    double check_sum = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++ ) {
            check_sum += kernel[i][j];
            cout<<kernel[i][j]<<" ";
        }
        cout<<endl;
    }
}

cv::Mat image_processing::gaussian_blur(cv::Mat input_img, double** kernel, uint32_t kernel_size) {

    int32_t k_neg_bound = kernel_size/2 * -1;
    int32_t k_pos_bound = kernel_size/2;

    cv::Mat output_img = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());

    for (int i = 0; i < input_img.rows; i++) {
        for (int j = 0; j < input_img.cols; j++) {
            for(int k_i = k_neg_bound; k_i <= k_pos_bound; k_i++) {
                for (int k_j = k_neg_bound; k_j <= k_pos_bound; k_j++) {
                    if (i + k_i >= 0 && i + k_i < input_img.rows && j + k_j >= 0 && j + k_j < input_img.cols) {
                        output_img.at<cv::Vec3b>(i,j)[0] += input_img.at<cv::Vec3b>(i + k_i, j + k_j)[0] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                        output_img.at<cv::Vec3b>(i,j)[1] += input_img.at<cv::Vec3b>(i + k_i, j + k_j)[1] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                        output_img.at<cv::Vec3b>(i,j)[2] += input_img.at<cv::Vec3b>(i + k_i, j + k_j)[2] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                    }
                }
            }
        }
    }
    return output_img;
}
