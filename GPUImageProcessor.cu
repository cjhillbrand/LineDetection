#include "GPUImageProcessor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int SIZE = 1024; // Number of threads per block (32 x 32)

// Kernel Functions...
__global__ void colorToGray(const uchar* input, uchar* d_gray, const int rowLength, const int colLength, const int colorFlag) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * rowLength + col;
    const int tid = threadIdx.x * blockDim.x + threadIdx.y;
   
    float grayval = 0;
    if (row < colLength && col < rowLength) {
        if (colorFlag == 1) {   
            d_gray[idx] += static_cast<uchar>(0.21 * input[idx]); //  Red
        }
        else if (colorFlag == 2) {
            d_gray[idx] += static_cast<uchar>(0.07 * input[idx]); //  Blue
        }
        else {
            d_gray[idx] += static_cast<uchar>(0.72 * input[idx]); //  Green
        }
    }
}

__constant__ int halfKernel = 1;

__global__ void convolution(const uchar* input, uchar* output, const uchar* kernel, const int kernelDim,
                             const int rows, const int cols, const float reduction, const int sobelFlag = 0, uchar* mask = 0) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x * blockDim.x + threadIdx.y;
    const int idx = row * cols + col;
    float sum = 0;

    if (row < rows && col < cols) {
        if (row < halfKernel || col < halfKernel || row + halfKernel > rows || col + halfKernel > cols) {
            // global edge case, do nothing for now
            output[idx] = input[idx];
        }
        else {
            // i iterates over the columns, j iterates over the rows in inner block
            if (kernelDim == 9 && !sobelFlag) {
                sum += input[idx - 1 * cols - 1] * kernel[(- 1 + halfKernel) * kernelDim + (- 1 + halfKernel)];  // -1 -1
                sum += input[idx * cols - 1] * kernel[(halfKernel) * kernelDim + (-1 + halfKernel)];             //  0 -1
                sum += input[idx + 1 * cols - 1] * kernel[(1 + halfKernel) * kernelDim + (-1 + halfKernel)];     //  1 -1
                sum += input[idx -1 * cols] * kernel[(-1 + halfKernel) * kernelDim + (halfKernel)];              // -1  0
                sum += input[idx * cols] * kernel[(halfKernel) * kernelDim + (halfKernel)];                      //  0  0
                sum += input[idx + 1 * cols] * kernel[(1 + halfKernel) * kernelDim + (halfKernel)];              //  1  0
                sum += input[idx - 1 * cols + 1] * kernel[(-1 + halfKernel) * kernelDim + (1 + halfKernel)];     // -1  1
                sum += input[idx * cols + 1] * kernel[(halfKernel) * kernelDim + (1 + halfKernel)];              //  0  1
                sum += input[idx + 1 * cols + 1] * kernel[(1 + halfKernel) * kernelDim + (1 + halfKernel)];      //  1  1
            }
            else if (sobelFlag) {
                if (sobelFlag == 1) {
                    sum -= input[idx - 1 * cols - 1] * kernel[(-1 + halfKernel) * kernelDim + (-1 + halfKernel)];
                } else {
                    sum += input[idx - 1 * cols - 1] * kernel[(-1 + halfKernel) * kernelDim + (-1 + halfKernel)];
                }
                if (sobelFlag == 2) {
                    sum += input[idx + 0 * cols - 1] * kernel[(halfKernel)*kernelDim + (-1 + halfKernel)];
                }
                sum += input[idx + 1 * cols - 1] * kernel[(1 + halfKernel) * kernelDim + (-1 + halfKernel)];   
                if (sobelFlag == 1) {
                    sum -= input[idx - 1 * cols] * kernel[(-1 + halfKernel) * kernelDim + (halfKernel)];
                }
                if (sobelFlag == 1) {
                    sum += input[idx + 1 * cols] * kernel[(1 + halfKernel) * kernelDim + (halfKernel)];
                }
                sum -= input[idx - 1 * cols + 1] * kernel[(-1 + halfKernel) * kernelDim + (1 + halfKernel)];
                if (sobelFlag == 2) {
                    sum -= input[idx + 0 * cols + 1] * kernel[(halfKernel)*kernelDim + (1 + halfKernel)];
                }
                if (sobelFlag == 2) {
                    sum -= input[idx + 1 * cols + 1] * kernel[(1 + halfKernel) * kernelDim + (1 + halfKernel)];
                } else {
                    sum += input[idx + 1 * cols + 1] * kernel[(1 + halfKernel) * kernelDim + (1 + halfKernel)];
                }
            }
            // TODO potentially remove this section
            else {
                for (int i = -halfKernel; i <= halfKernel; i++) {
                    for (int j = -halfKernel; j <= halfKernel; j++) {
                        // Gx kernel when sobel == 1, Gy kernel when sobel == 2
                        if ((sobelFlag == 1 && j == -halfKernel) || (sobelFlag == 2 && i == halfKernel)) {
                            sum -= input[idx + i * cols + j] * kernel[(i + halfKernel) * kernelDim + (j + halfKernel)];
                        }
                        else {
                            sum += input[idx + i * cols + j] * kernel[(i + halfKernel) * kernelDim + (j + halfKernel)];
                        }
                    }
                }
            }
            if (sobelFlag) {
                mask[idx] = sum < 0 ? 1 : 0;
            }
            output[idx] = static_cast<uchar>(sum / reduction);
        }
    }
}

__global__ void threshold(const uchar* input, uchar* output, const int rows, const int cols, const int lowerBound) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * cols + col;
    const int MAX_UCHAR = 255;
    const int MIN_UCHAR = 0;

    if (row < rows && col < cols) {
        output[idx] = input[idx] >= lowerBound ? MAX_UCHAR : MIN_UCHAR;
    }
}

__global__ void sobelGradientMagnitude(uchar* output, const uchar* outGx, const uchar* outGy, const uchar* maskGx, const uchar* maskGy,
                    const int rows, const int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * cols + col;
    const int tid = threadIdx.x * blockDim.x + threadIdx.y;
    
    __shared__ uchar outputGx[SIZE];
    __shared__ uchar outputGy[SIZE];
    
    outputGx[tid] = outGx[idx];
    outputGy[tid] = outGy[idx];

    __syncthreads();
    
    if (row < rows && col < cols) {
        
        output[idx] = static_cast<uchar>(sqrt(pow(maskGx[idx]
            == 1 ? -static_cast<float>(outputGx[tid]) : static_cast<float>(outputGx[tid]), 2)
            + pow(maskGy[idx] == 1 ? -static_cast<float>(outputGy[tid]) : static_cast<float>(outputGy[tid]), 2)));
    }
}

__global__ void TCTOA(const uchar* bgrComponent, uchar* output, const int rows, const int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = row * cols + col;
    const int tid = threadIdx.x * blockDim.x + threadIdx.y;

    __shared__ uchar bgr[SIZE];
    bgr[tid] = bgrComponent[idx];

    __syncthreads();

    output[idx] = bgr[tid];
}

GPUImageProcessor::GPUImageProcessor(std::string filename) : ImageProcessor(filename ) {}

void GPUImageProcessor::preProcess(const Mat& frame, Mat& output) {  
   
    Mat gray(frame.rows, frame.cols, CV_8UC1);
    Mat smooth(frame.rows, frame.cols, CV_8UC1);
    Mat thresh(frame.rows, frame.cols, CV_8UC1);
    
    GPUImageGray(frame, gray);
    GPUImageSmooth(gray, smooth);
    GPUImageThreshold(smooth, thresh);
    GPUImageEdge(thresh, output);
}

void GPUImageProcessor::GPUImageThreshold(const Mat& input, Mat& output) {
    const int rows = input.rows, cols = input.cols;
    uchar* d_input, * d_output;
    const int size = rows * cols;
    const dim3 blocks(ceil((double)cols / 32.0), ceil((double)rows / 32.0));
    const dim3 threads(32, 32);

    //220
    const uchar lowerBound = 230;

    cudaMalloc<unsigned char>(&d_input, size);
    cudaMalloc<unsigned char>(&d_output, size);

    cudaMemcpyAsync(d_input, input.ptr(), size, cudaMemcpyHostToDevice);

    threshold << <blocks, threads >> > (d_input, d_output, rows, cols, lowerBound);
    cudaDeviceSynchronize();

    cudaMemcpy(output.ptr(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

void GPUImageProcessor::GPUImageEdge(const Mat& input, Mat& output) {
    const int rows = input.rows, cols = input.cols;
    uchar* d_input, *d_output, *d_kernelGx, *d_kernelGy, * d_maskGx, *d_maskGy, *d_outputGx, *d_outputGy;
    const int size = rows * cols;
    const int reduction = 1;

    uchar Gx[9] = { 1, 0, 1, 2, 0, 2, 1, 0, 1 };
    uchar Gy[9] = { 1, 2, 1, 0, 0, 0, 1, 2, 1 };

    Mat maskGx(rows, cols, CV_8UC1);
    Mat maskGy(rows, cols, CV_8UC1);
    Mat outputGx(rows, cols, CV_8UC1);
    Mat outputGy(rows, cols, CV_8UC1);
    Mat kernelGx(3, 3, CV_8UC1, Gx);
    Mat kernelGy(3, 3, CV_8UC1, Gy);

    const dim3 blocks(ceil((double)cols / 32.0), ceil((double)rows / 32.0));
    const dim3 threads(32, 32);

    const int NUM_OF_STREAMS = 2;
    cudaStream_t streams[NUM_OF_STREAMS];
   
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Gx
    cudaMalloc<unsigned char>(&d_input, size);
    cudaMalloc<unsigned char>(&d_kernelGx, 9);
    cudaMalloc<unsigned char>(&d_maskGx, size);
    cudaMalloc<unsigned char>(&d_outputGx, size);
    cudaMemcpyAsync(d_input, input.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_kernelGx, kernelGx.ptr(), 9, cudaMemcpyHostToDevice);
    
    convolution << <blocks, threads, 0, streams[0] >> > (d_input, d_outputGx, d_kernelGx, 3, rows, cols, reduction, 1, d_maskGx);

    cudaMemcpyAsync(outputGx.ptr(), d_outputGx, size, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(maskGx.ptr(), d_maskGx, size, cudaMemcpyDeviceToHost);

    cudaFree(d_kernelGx);

    // Gy
    cudaMalloc<unsigned char>(&d_kernelGy, 9);
    cudaMalloc<unsigned char>(&d_maskGy, size);
    cudaMalloc<unsigned char>(&d_outputGy, size);

    cudaMemcpyAsync(d_kernelGy, kernelGy.ptr(), 9, cudaMemcpyHostToDevice);
 

    convolution <<<blocks, threads, 0, streams[1] >>> (d_input, d_outputGy, d_kernelGy, 3, rows, cols, reduction, 2, d_maskGy);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(outputGy.ptr(), d_outputGy, size, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(maskGy.ptr(), d_maskGy, size, cudaMemcpyDeviceToHost);

    cudaFree(d_kernelGy);
    cudaFree(d_input);

    cudaMalloc(&d_output, size);

    // sqrt(Gx^2 + Gy^2
    sobelGradientMagnitude << <blocks, threads >> > (d_output, d_outputGx, d_outputGy, d_maskGx, d_maskGy, rows, cols);
    cudaDeviceSynchronize();


    cudaMemcpy(output.ptr(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_outputGx);
    cudaFree(d_outputGy);
    cudaFree(d_maskGy);
    cudaFree(d_maskGx);
    cudaFree(d_input);
}

void GPUImageProcessor::GPUImageGray(const Mat& input, Mat& output) {
    const int rows = input.rows, cols = input.cols;
    Mat red(rows, cols, CV_8UC1), green(rows, cols, CV_8UC1), blue(rows, cols, CV_8UC1);
    uchar* d_red, * d_green, * d_blue, * d_gray;

    const int size = red.step * red.rows;
    const int NUM_OF_STREAMS = 3;
    cudaStream_t streams[NUM_OF_STREAMS];

    dim3 blocks(ceil((double)cols / 32.0), ceil((double)rows / 32.0));
    dim3 threads(32, 32);

    threeChannelToArray(input, red, green, blue);

    cudaMalloc<unsigned char>(&d_gray, size);
    cudaMalloc<unsigned char>(&d_red, size);
    cudaMalloc<unsigned char>(&d_green, size);
    cudaMalloc<unsigned char>(&d_blue, size);

    cudaMemcpyAsync(d_red, red.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_green, green.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_blue, blue.ptr(), size, cudaMemcpyHostToDevice);

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);

    colorToGray << <blocks, threads, 0, streams[0] >> > (d_red, d_gray, red.cols, red.rows, 1);
    colorToGray << <blocks, threads, 0, streams[1] >> > (d_green, d_gray, red.cols, red.rows, 2);
    colorToGray << <blocks, threads, 0, streams[2] >> > (d_blue, d_gray, red.cols, red.rows, 3);
    cudaDeviceSynchronize();
    
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

    cudaMemcpyAsync(d_input, input.ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_kernel, kernel.ptr(), 9, cudaMemcpyHostToDevice);

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

    const int size = frame.rows * frame.cols;
    const int rows = frame.rows, cols = frame.cols;

    const dim3 blocks(ceil((double)cols / 32.0), ceil((double)rows / 32.0));
    const dim3 threads(32, 32);

    const int NUM_OF_STREAMS = 3;
    cudaStream_t streams[NUM_OF_STREAMS];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);

    split(frame, bgrComponents);
 
    uchar *d_redComponent, *d_blueComponent, *d_greenComponent, *d_red, *d_blue, *d_green;

    cudaMalloc<unsigned char>(&d_red, size);
    cudaMalloc<unsigned char>(&d_blue, size);
    cudaMalloc<unsigned char>(&d_green, size);
    cudaMalloc<unsigned char>(&d_redComponent, size);
    cudaMalloc<unsigned char>(&d_blueComponent, size);
    cudaMalloc<unsigned char>(&d_greenComponent, size);

   
    cudaMemcpyAsync(d_redComponent, bgrComponents[0].ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_blueComponent, bgrComponents[1].ptr(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_greenComponent, bgrComponents[2].ptr(), size, cudaMemcpyHostToDevice);

    TCTOA << <blocks, threads, 0, streams[0] >> > (d_redComponent, d_red, rows, cols);
    TCTOA << <blocks, threads, 0, streams[1] >> > (d_blueComponent, d_blue, rows, cols);
    TCTOA << <blocks, threads, 0, streams[2] >> > (d_greenComponent, d_green, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(red.ptr(), d_red, size, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(blue.ptr(), d_blue, size, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(green.ptr(), d_green, size, cudaMemcpyDeviceToHost);

    cudaFree(d_red);
    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_redComponent);
    cudaFree(d_blueComponent);
    cudaFree(d_greenComponent);
    
}

