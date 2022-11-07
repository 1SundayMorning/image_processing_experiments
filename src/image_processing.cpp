#include "image_processing.h"

#define PI 3.14159

void image_processing::hough_circle_detection(cv::Mat input_img, bool debug) {
    debug = true;
    if (debug) cv::imshow("input_image", input_img);

    // gaussian blur image
    double sigma = 3;
    uint32_t kernel_size = 7;
    cv::Mat gauss_img = image_processing::gaussian_blur(input_img, sigma, kernel_size);
    if (debug) cv::imshow("gauss_image", gauss_img);

    // convert to greyscale
    cv::Mat grey_img;
    cv::cvtColor(gauss_img, grey_img, cv::COLOR_BGR2GRAY);
    if (debug) cv::imshow("grey_image", grey_img);

    // convert to binary image using thresholding
    cv::Mat binary_img = image_processing::convert_binary_image(grey_img);
    if (debug) cv::imshow("binary image", binary_img);

    // laplacian convolution
    cv::Mat laplacian_img = image_processing::conv_laplacian(binary_img);

    // sobel edge detection
    cv::Mat sobel_img = sobel_edge(binary_img); 
    if (debug) cv::imshow("sobel image", sobel_img);


}

cv::Mat image_processing::pad_image(cv::Mat input_img, uint32_t pad_height, uint32_t pad_width) {
    uint32_t image_height = input_img.rows;
    uint32_t image_width = input_img.cols;
    cv::Mat padded_img = cv::Mat::zeros(image_height + 2 * pad_height, image_width + 2 * pad_width, input_img.type());

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

cv::Mat image_processing::conv_image_2d_3chan(cv::Mat input_img, vector<vector<int>> kernel, int kernel_size, bool padding) {

    cv::Mat output_img = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
    
    int k_pos_bound = kernel_size/2;
    int k_neg_bound = k_pos_bound * -1;

    for (int i = 0; i < input_img.rows; i++) {
        for (int j = 0; j < input_img.cols; j++) {
            for(int k_i = k_neg_bound; k_i <= k_pos_bound; k_i++) {
                for (int k_j = k_neg_bound; k_j <= k_pos_bound; k_j++) {
                    if (padding) {
                        int i_adj = i + k_pos_bound;
                        int j_adj = j + k_pos_bound;
                        output_img.at<cv::Vec3b>(i_adj,j_adj)[0] += input_img.at<cv::Vec3b>(i_adj + k_i, j_adj + k_j)[0] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                        output_img.at<cv::Vec3b>(i_adj,j_adj)[1] += input_img.at<cv::Vec3b>(i_adj + k_i, j_adj + k_j)[1] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                        output_img.at<cv::Vec3b>(i_adj,j_adj)[2] += input_img.at<cv::Vec3b>(i_adj + k_i, j_adj + k_j)[2] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                    }
                    else {
                        if (i + k_i >= 0 && i + k_i < input_img.rows && j + k_j >= 0 && j + k_j < input_img.cols) {
                            output_img.at<cv::Vec3b>(i,j)[0] += input_img.at<cv::Vec3b>(i + k_i, j + k_j)[0] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                            output_img.at<cv::Vec3b>(i,j)[1] += input_img.at<cv::Vec3b>(i + k_i, j + k_j)[1] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                            output_img.at<cv::Vec3b>(i,j)[2] += input_img.at<cv::Vec3b>(i + k_i, j + k_j)[2] * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                        }
                    }
                }
            }
        }
    }
    return output_img;
}

cv::Mat image_processing::conv_image_2d_1chan(cv::Mat input_img, vector<vector<int>> kernel, int kernel_size, bool padding) {

    cv::Mat output_img = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
    
    int k_pos_bound = kernel_size/2;
    int k_neg_bound = k_pos_bound * -1;

    for (int i = 0; i < input_img.rows; i++) {
        for (int j = 0; j < input_img.cols; j++) {
            for(int k_i = k_neg_bound; k_i <= k_pos_bound; k_i++) {
                for (int k_j = k_neg_bound; k_j <= k_pos_bound; k_j++) {
                    if (padding) {
                        int i_adj = i + k_pos_bound;
                        int j_adj = j + k_pos_bound;
                        output_img.at<uchar>(i_adj,j_adj) += input_img.at<uchar>(i_adj + k_i, j_adj + k_j) * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                    }
                    else {
                        if (i + k_i >= 0 && i + k_i < input_img.rows && j + k_j >= 0 && j + k_j < input_img.cols) {
                            output_img.at<uchar>(i,j) += input_img.at<uchar>(i + k_i, j + k_j) * kernel[k_i + k_pos_bound][k_j + k_pos_bound];
                        }
                    }
                }
            }
        }
    }
    return output_img;
}

cv::Mat image_processing::conv_laplacian(cv::Mat input_img) {
    int8_t laplacian_kernel[3][3] = {{-1, -1, -1},{-1, 8, -1},{-1, -1, -1}};
    // laplacian_kernel = {{0, -1, 0},{-1, 4, -1},{0, -1, 0}};
    cv::Mat result;
    return result;
}

cv::Mat image_processing::sobel_edge(cv::Mat input_img) {
    vector<vector<int>> sobel_x_kernel = {{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}};
    vector<vector<int>> sobel_y_kernel = {{1, 2, 1},{0, 0, 0},{-1, -2, -1}};

    cv::Mat padded_img = image_processing::pad_image(input_img, 1, 1);

    cv::Mat xconv_img = image_processing::conv_image_2d_1chan(padded_img, sobel_x_kernel, 3, true);
    cv::Mat yconv_img = image_processing::conv_image_2d_1chan(padded_img, sobel_y_kernel, 3, true);

    cv::Mat result = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());

    int max_val = -1;
    int min_val = 1000000;

    for(int i = 0; i < input_img.rows; i++) {
        for (int j = 0; j < input_img.cols; j++) {
            int convx = xconv_img.at<uchar>(i,j);
            int convy = yconv_img.at<uchar>(i,j);
            int result_val = sqrt(convx * convx + convy * convy);
            result.at<uchar>(i,j) = result_val;

            if (result_val < min_val) min_val = result_val;
            if (result_val > max_val) max_val = result_val;
        }
    }

    int min_max_differential = max_val - min_val;

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            float ratio = ((float)result.at<uchar>(i,j) - min_val) / (float)(min_max_differential);
            result.at<uchar>(i,j) = ratio * 255;
        }
    }


    return result;
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

cv::Mat image_processing::convert_binary_image(cv::Mat input_img) {
    if (input_img.channels() > 1) {
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2GRAY);
    }

    uint32_t thresh_setpoint = get_average_pixel_intensity(input_img);

    for (int i = 0; i < input_img.rows; i++) {
        for (int j = 0; j < input_img.cols; j++) {
            if ((uint32_t)input_img.at<uchar>(i,j) < thresh_setpoint) {
                input_img.at<uchar>(i,j) = 0;
            }
            else {
                input_img.at<uchar>(i,j) = 255;
            }
        }
    }
    return input_img;
}

cv::Mat image_processing::convert_binary_image(cv::Mat input_img, uint32_t setpoint) {
    if (input_img.channels() > 1) {
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2GRAY);
    }

    for (int i = 0; i < input_img.rows; i++) {
        for (int j = 0; j < input_img.cols; j++) {
            if ((uint32_t)input_img.at<uchar>(i,j) < setpoint) {
                input_img.at<uchar>(i,j) = 0;
            }
            else {
                input_img.at<uchar>(i,j) = 255;
            }
        }
    }
    return input_img;
}

cv::Mat image_processing::gaussian_blur(cv::Mat input_img, double sigma, uint32_t kernel_size) {
    double ** gauss_kernel;
    gauss_kernel = new double*[kernel_size];
    for (int i = 0; i < kernel_size; i++) {
        gauss_kernel[i] = new double[kernel_size];
    }
    image_processing::create_gaussian_kernel(sigma, gauss_kernel, kernel_size);
    cv::Mat gauss_img = image_processing::conv_gaussian(input_img, gauss_kernel, kernel_size);

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
    // for (int i = 0; i < kernel_size; i++) {
    //     for (int j = 0; j < kernel_size; j++ ) {
    //         check_sum += kernel[i][j];
    //         cout<<kernel[i][j]<<" ";
    //     }
    //     cout<<endl;
    // }
}

cv::Mat image_processing::conv_gaussian(cv::Mat input_img, double** kernel, uint32_t kernel_size) {

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
