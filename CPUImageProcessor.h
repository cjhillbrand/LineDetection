#include "ImageProcessor.h"

#ifndef CPU_IMAGE_PROCESSOR
#define CPU_IMAGE_PROCESSOR
class CPUImageProcessor {
    public:
	override void preProcess(const Mat&, Mat&);
	override void houghLineTransform(const Mat&, Mat&);
}



#endif
