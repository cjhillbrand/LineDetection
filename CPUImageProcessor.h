#include "ImageProcessor.h"
#include <vector>
using namespace std;
using namespace cv;

#ifndef CPU_IMAGE_PROCESSOR
#define CPU_IMAGE_PROCESSOR
class CPUImageProcessor : public ImageProcessor {
    public:
	CPUImageProcessor();
	void preProcess(const Mat&, Mat&) override; 
	void houghLineTransform(const Mat&, Mat&) override;
};



#endif
