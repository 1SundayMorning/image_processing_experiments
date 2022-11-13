#include "image_processing.h"

#define PI 3.14159

vector<vector<vector<int>>> image_processing::hough_circle_detection(cv::Mat input_img, float threshold, int region, int min_radius, int max_radius, bool debug) {
    // gaussian blur image
    double sigma = 3;
    uint32_t kernel_size = 7;
    cv::Mat gauss_img = image_processing::gaussian_blur(input_img, sigma, kernel_size);

    // convert to greyscale
    cv::Mat grey_img;
    cv::cvtColor(gauss_img, grey_img, cv::COLOR_BGR2GRAY);

    // convert to binary image using thresholding
    cv::Mat binary_img = image_processing::convert_binary_image(grey_img);

    // laplacian convolution
    cv::Mat laplacian_img = image_processing::laplacian_edge(binary_img);
    cv::imshow("laplacian image: ", laplacian_img);

    int image_rows = input_img.rows;
    int image_cols = input_img.cols;

    double cos_thetas[360];
    double sin_thetas[360];
    for (int i = 0; i < 360; i++) {
        cos_thetas[i] = cos(i * PI/180);
        sin_thetas[i] = sin(i * PI/180);
    }

    vector<vector<int>> circles;
    for (int r = min_radius; r <= max_radius; r++) {
        for (int theta = 0; theta < 360; theta++) {
            circles.push_back({r, static_cast<int>(r * cos_thetas[theta]), static_cast<int>(r * sin_thetas[theta])});

        }
    }

    vector<vector<int>> edges;
    for (int i = 0; i < laplacian_img.rows; i++) {
        for (int j = 0; j < laplacian_img.cols; j++) {
            if (laplacian_img.at<uchar>(i,j)) {
                edges.push_back({i, j});
            }
        }
    }
    cout<<"num edge points: "<<edges.size()<<endl;

    cout<<"circles size: "<<circles.size()<<endl;

    // padded with the diameter of the maximum radius
    vector<vector<vector<int>>> Accumulator(max_radius, vector<vector<int>>(max_radius * 2 + input_img.rows, vector<int>(max_radius * 2 + input_img.cols, 0)));
    vector<vector<vector<int>>> Result(max_radius, vector<vector<int>>(max_radius * 2 + input_img.rows, vector<int>(max_radius * 2 + input_img.cols, 0)));
    for (int r = min_radius; r < max_radius; r++) {
        // circle blueprint
        vector<vector<int>> blueprint(2 * (r + 1), vector<int>(2 * (r + 1), 0));
        vector<int> blueprint_center = {r + 1, r + 1};
        for (int t = 0; t < 360; t++) {
            int circle_x_coord = blueprint_center[0] + (int)(r * cos_thetas[t]);
            int circle_y_coord = blueprint_center[1] + (int)(r * sin_thetas[t]);
            blueprint[circle_x_coord][circle_y_coord] = 1;
        }
        int circle_pt_count = 360;

        // cout<<"edges x size: "<<edges.size()<<endl;
        // cout<<"edges y size: "<<edges[0].size()<<endl;
        // cout<<"loop thru edges"<<endl;
        for (int i = 0; i < edges.size(); i++) {
            int X_0 = edges[i][0] - blueprint_center[0] + max_radius;
            int X_1 = edges[i][0] + blueprint_center[0] + max_radius;
            int Y_0 = edges[i][1] - blueprint_center[1] + max_radius;
            int Y_1 = edges[i][1] + blueprint_center[1] + max_radius;
            // if(X_0 > 400 || X_1 > 400) {
            //     cout<<"X_0: "<<X_0<<endl;
            //     cout<<"X_1: "<<X_1<<endl;
            // }
            // if(Y_0 > 400 || Y_1 > 400) {
            //     cout<<"Y_0: "<<Y_0<<endl;
            //     cout<<"Y_1: "<<Y_1<<endl;
            // }
            // cout<<"Y_0: "<<Y_0<<endl;
            // cout<<"Y_1: "<<Y_1<<endl;
            // if (X_0 >= image_cols || X_1 >= image_cols || Y_0 >= image_rows || Y_1 >= image_rows) {
            //     continue;
            // }
            vector<int> X = {X_0, X_1};
            vector<int> Y = {Y_0, Y_1};
            // cout<<"add blueprint pts"<<endl;
            for (int x = 0; x < X[1] - X[0]; x++) {
                for (int y = 0; y < Y[1] - Y[0]; y++) {
                    Accumulator[r][x + X[0]][y + Y[0]] += blueprint[x][y];
                }
            }
        }
        // cout<<"filter Accumulator"<<endl;
        for (int x = 0; x < Accumulator[r].size(); x++) {
            for (int y = 0; y < Accumulator[r][x].size(); y++) {
                if (Accumulator[r][x][y] < threshold * circle_pt_count / r) {
                    Accumulator[r][x][y] = 0;
                }
            }
        }
    }
    // cout<<Accumulator[0].size()/region<<endl;
    // cout<<Accumulator[0][0].size()/region<<endl;
    cout<<"start build result matrix"<<endl;
    for (int r = min_radius; r < max_radius; r++) {
        for (int x_block = 0; x_block < Accumulator[r].size(); x_block += region) {
            for (int y_block = 0; y_block < Accumulator[r][0].size(); y_block += region) {
                int maximum = 0;
                for (int x = x_block; x < x_block + region && x < Accumulator[r].size(); x++) {
                    for (int y = y_block; y < y_block + region && y < Accumulator[r][0].size(); y++) {
                        // if (y > 500) {
                        //     cout<<"Y > 500: "<<y<<endl;
                        // }
                        if (Accumulator[r][x][y] > maximum) {
                            maximum = Accumulator[r][x][y];
                            Result[r][x][y] = 1;

                        }
                    }
                }
            }
        }
    }

    cout<<"End of hough"<<endl;
    return Result;
}

cv::Mat image_processing::draw_circles(cv::Mat input_image, vector<vector<vector<int>>> input_circles) {
    cv::Scalar line_color(255, 0, 0);//Color of the circle
    int thickness = 2;

    cv::Mat result_img = input_image.clone();

    for(int r = 0; r < input_circles.size(); r++) {
        for (int x = 0; x < input_circles[r].size(); x++) {
            for (int y = 0; y < input_circles[r][x].size(); y++) {
                if (input_circles[r][x][y]) {
                    cv::Point center(y,x);
                    cv::circle(result_img, center, r, line_color, thickness);
                }
            }
        }
    }
    return result_img;
}

cv::Mat image_processing::opencv_hough(cv::Mat input_img) {
    cv::Mat grey_img;
    cv::cvtColor(input_img, grey_img, cv::COLOR_BGR2GRAY);
    cv::imshow("Grey", grey_img);
    cv::medianBlur(grey_img, grey_img, 5);
    cv::imshow("grey blurred", grey_img);
    vector<cv::Vec3f> circles;
    HoughCircles(grey_img, circles, cv::HOUGH_GRADIENT, 1,
                 grey_img.rows/16,
                 100, 30, 30, 50
    );
    cout<<"hough complete"<<endl;
    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle center
        cv::circle(input_img, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
        // circle outline
        int radius = c[2];
        circle(input_img, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
    }
    return input_img;
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

cv::Mat image_processing::laplacian_edge(cv::Mat input_img) {
    vector<vector<int>> laplacian_kernel = {{-1, -1, -1},{-1, 8, -1},{-1, -1, -1}};

    cv::Mat padded_img = image_processing::pad_image(input_img, 1, 1);
    
    cv::Mat result = image_processing::conv_image_2d_1chan(padded_img, laplacian_kernel, 3, true);
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
