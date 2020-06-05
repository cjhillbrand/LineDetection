#include "ImageProcessor.h"
#include <thread>

#ifndef GPU_IMAGE_PROCESSOR
#define GPU_IMAGE_PROCESSOR
class GPUImageProcessor : public ImageProcessor {
public:
    GPUImageProcessor(std::string);
    void preProcess(const Mat&, Mat&) override;
    void houghLineTransform(Mat&, Mat&) override;
private:
    //    	__device__ void colorToGray(const float*, const float*, const float*, float*, const int);
        //	__kernel__ smoothing
        //	__kernel__ threshold
        //	if we get to it __kernel__ canny
        //
        //	mattoarray helper methods
        //	arraytomat helper methods
        //
    void threeChannelToArray(const Mat&, Mat&, Mat&, Mat&);
    void GPUImageSmooth(const Mat& input, Mat& output);
    void GPUImageGray(const Mat& input, Mat& output);
    void GPUImageEdge(const Mat& input, Mat& output);
    void GPUImageThreshold(const Mat& input, Mat& output);
};

#endif
