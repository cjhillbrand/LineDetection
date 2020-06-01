#include "GPUImageProcessor.h"
GPUImageProcessor::GPUImageProcessor() {
    printf("Not implemented yet");
}

void GPUImageProcessor::preProcess(const Mat& frame, Mat& result) {
    printf("Not implemented yet");
}

void GPUImageProcessor::houghLineTransform(Mat& frame, Mat& result) {
    printf("Not Implemented yet");
}

void GPUImageProcessor::threeChannelToArray(const Mat& frame, float* red, float* blue, float* green) {
    uint8_t* matPtr = (uint8_t*) frame.data;
    int cn = frame.channels();

    const int rows = frame.rows;
    const int cols = frame.cols;

    // Iterate over the mat
    for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
	    blue[i * rows + j] = matPtr[i * cols * cn + j * cn + 0]; // B
	    green[i * rows + j] = matPtr[i * cols * cn + j * cn + 1]; // G
	    red[i * rows + j] = matPtr[i * cols * cn + j * cn + 2]; // R
	}
    }
}

// Kernel Functions...
__global__ void colorToGray(const float* red, const float* blue, const float* green, float* gray, const int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * n + col;

    gray[idx] = 0.21 * red[idx] + 0.07 * blue[idx] + 0.72 * green[idx];
}
