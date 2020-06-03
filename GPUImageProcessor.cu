#include "GPUImageProcessor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Kernel Functions...
__global__ void colorToGray(const uchar* red, const uchar* blue, const uchar* green,
    uchar* d_gray, const int rowLength, const int colLength) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * rowLength + col;

    float grayval = 0;
    if (row < colLength && col < rowLength) {
        grayval = 0.21 * red[idx] + 0.07 * blue[idx] + 0.72 * green[idx];
        d_gray[idx] = static_cast<uchar>(grayval);
        //d_gray[idx] = 0;
    }
}

__global__ void convolution(const uchar* input, uchar* output, const uchar* kernel, const int kernelDim, 
                                    const int rows, const int cols, const float reduction, const bool sobel = false) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * cols + col;

    if (row < rows && col < cols) {
        if (row < kernelDim / 2 || col < kernelDim / 2 || row + kernelDim / 2 > rows || col + kernelDim / 2 > cols) {
            // edge case, do nothing for now
            output[idx] = input[idx];
        }
        else {
            float sum = 0;
            for (int i = -kernelDim / 2; i <= kernelDim / 2; i++) {
                for (int j = -kernelDim / 2; j <= kernelDim / 2; j++) {
                    // i iterates over the columns, j iterates over the rows
                    if (i != 0 && j != 0 && sobel) {
                        sum -= input[idx + i * cols + j] * kernel[(i + kernelDim / 2) * kernelDim + (j + kernelDim / 2)];
                    }
                    else {
                        sum += input[idx + i * cols + j] * kernel[(i + kernelDim / 2) * kernelDim + (j + kernelDim / 2)];
                    }
                }
            }
            output[idx] = static_cast<uchar>(sum / reduction);
        }
    }
    // do nothing, thread is out of bounds
}

GPUImageProcessor::GPUImageProcessor() {
    printf("Not implemented yet");
}

void GPUImageProcessor::preProcess(const Mat& frame, Mat& output) {
    Mat gray(frame.rows, frame.cols, CV_8UC1);
    Mat smooth(frame.rows, frame.cols, CV_8UC1);

    GPUImageGray(frame, gray);
    imshow("gray", gray);
    waitKey(0);

    GPUImageSmooth(gray, smooth);
    imshow("gray and smooth", smooth);
    waitKey(0);

    GPUImageEdge(smooth, output);
    imshow("edge image", output);
    waitKey(0);
    
}

void GPUImageProcessor::GPUImageEdge(const Mat& input, Mat& output) {
    const int rows = input.rows, cols = input.cols;
    uchar* d_input, * d_output, *d_kernel;
    const int size = rows * cols;
    const int reduction = 1;

    cudaMalloc<unsigned char>(&d_input, size);
    cudaMalloc<unsigned char>(&d_output, size);
    cudaMalloc<unsigned char>(&d_kernel, 9);

    uchar data[9] = { 1, 1, 1, 1, 8, 1, 1, 1, 1 };
    Mat kernel(3, 3, CV_8UC1, data);

    cudaMemcpy(d_input, input.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.ptr(), 9, cudaMemcpyHostToDevice);


    const dim3 blocks(ceil((double)cols / 32.0), ceil((double)rows / 32.0));
    const dim3 threads(32, 32);

    convolution << <blocks, threads >> > (d_input, d_output, d_kernel, 3, rows, cols, reduction, true);
    cudaDeviceSynchronize();

    cudaMemcpy(output.ptr(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}




void GPUImageProcessor::GPUImageGray(const Mat& input, Mat& output) {
    const int rows = input.rows, cols = input.cols;
    Mat red(rows, cols, CV_8UC1), green(rows, cols, CV_8UC1), blue(rows, cols, CV_8UC1);
    uchar* d_red, * d_green, * d_blue, * d_gray;

    const int size = red.step * red.rows;

    dim3 blocks(ceil((double)cols / 32.0), ceil((double)rows / 32.0));
    dim3 threads(32, 32);

    cudaMalloc<unsigned char>(&d_red, size);
    cudaMalloc<unsigned char>(&d_green, size);
    cudaMalloc<unsigned char>(&d_blue, size);
    cudaMalloc<unsigned char>(&d_gray, size);

    threeChannelToArray(input, red, green, blue);


    cudaMemcpy(d_red, red.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, blue.ptr(), size, cudaMemcpyHostToDevice);

    colorToGray << <blocks, threads >> > (d_red, d_blue, d_green, d_gray, red.cols, red.rows);
    cudaDeviceSynchronize();
    std::cout << "Here" << std::endl;

    cudaMemcpy(output.ptr(), d_gray, size, cudaMemcpyDeviceToHost);
    cudaFree(d_red);
    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_gray);
}
void GPUImageProcessor::GPUImageSmooth(const Mat& input, Mat& output) {
    const int rows = input.rows, cols = input.cols;
    const int size = rows * cols;
    
    uchar data[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    Mat kernel(3, 3, CV_8UC1, data);
    const float reduction = 16;
    uchar* d_input, * d_output, * d_kernel;

    cudaMalloc<unsigned char>(&d_input, size);
    cudaMalloc<unsigned char>(&d_output, size);
    cudaMalloc<unsigned char>(&d_kernel, 9);


    cudaMemcpy(d_input, input.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.ptr(), 9, cudaMemcpyHostToDevice);

    const dim3 blocks(ceil((double)cols / 32.0), ceil((double)rows / 32.0));
    const dim3 threads(32, 32);

    convolution << <blocks, threads >> > (d_input, d_output, d_kernel, 3, rows, cols, reduction);
    cudaDeviceSynchronize();

    cudaMemcpy(output.ptr(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

void GPUImageProcessor::houghLineTransform(Mat& frame, Mat& result) {
    printf("Not Implemented yet");
}

void GPUImageProcessor::threeChannelToArray(const Mat& frame, Mat& red, Mat& blue, Mat& green) {
    Mat bgrComponents[3];
    split(frame, bgrComponents);
    //red = bgrComponents[0].ptr();
    //blue = bgrComponents[1].ptr();
    //green = bgrComponents[2].ptr();

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            red.at<uchar>(i, j) = bgrComponents[0].at<uchar>(i,j);
            blue.at<uchar>(i, j) = bgrComponents[1].at<uchar>(i, j);
            green.at<uchar>(i, j) = bgrComponents[2].at<uchar>(i, j);
        }
    }
}

