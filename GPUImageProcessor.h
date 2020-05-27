#include "ImageProcessor.h"

#ifndef GPU_IMAGE_PROCESSOR
#define GPU_IMAGE_PROCESSOR
class GPUImageProcessor {
    public:
	override void preProcess(const Mat&, Mat&);
	override void houghLineTransform(const Mat&, Mat&);
}



#endif
