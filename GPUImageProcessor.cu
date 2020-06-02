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

GPUImageProcessor::GPUImageProcessor() {
    printf("Not implemented yet");
}

void GPUImageProcessor::preProcess(const Mat& frame, Mat& result) {
    const int rows = frame.rows, cols = frame.cols;
    Mat red(rows, cols, CV_8UC1), green(rows, cols, CV_8UC1), blue(rows, cols, CV_8UC1), gray(rows, cols, CV_8UC1);
    uchar* d_red, * d_green, * d_blue, *d_gray;

    int size = red.step * red.rows;
    const int THREAD_DIM = 32;// TODO
    int NUM_OF_BLOCKS = frame.rows / 32; // TODO

    dim3 blocks(ceil((double)frame.cols/32.0), ceil((double)frame.rows/32.0));
    dim3 threads(32, 32);

    cudaMalloc<unsigned char>(&d_red, size);
    cudaMalloc<unsigned char>(&d_green, size);
    cudaMalloc<unsigned char>(&d_blue, size);
    cudaMalloc<unsigned char>(&d_gray, size);

    threeChannelToArray(frame, red, green, blue);

    cudaMemcpy(d_red, red.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, blue.ptr(), size, cudaMemcpyHostToDevice);
 
    colorToGray<<<blocks, threads >>>(d_red, d_blue, d_green, d_gray, red.cols, red.rows);
    cudaDeviceSynchronize();
    cudaMemcpy(gray.ptr(), d_gray, size, cudaMemcpyDeviceToHost);

    imshow("Gray image", gray);
    waitKey(0);

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

