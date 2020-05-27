#include "ImageProcessor.h"

#ifndef CPU_IMAGE_PROCESSOR
#define CPU_IMAGE_PROCESSOR
class CPUImageProcessor : public ImageProcessor {
    public:
	CPUImageProcessor();
	void preProcess(const Mat&, Mat&) override; 
	void houghLineTransform(const Mat&, Mat&) override;
};



#endif
