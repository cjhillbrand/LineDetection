#include "ImageProcessor.h"

#ifndef GPU_IMAGE_PROCESSOR
#define GPU_IMAGE_PROCESSOR
class GPUImageProcessor : public ImageProcessor {
    public:
	GPUImageProcessor();
	void preProcess(const Mat&, Mat&) override;
	void houghLineTransform(Mat&, Mat&) override;
};



#endif
