#include "ImageProcessor.h"

#ifndef GPU_IMAGE_PROCESSOR
#define GPU_IMAGE_PROCESSOR
class GPUImageProcessor : public ImageProcessor {
    public:
	GPUImageProcessor();
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
    	void threeChannelToArray(const Mat&, float*, float*, float*);
};



#endif
