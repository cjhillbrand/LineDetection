#include <opencv2/core.hpp>
using namespace cv;

#ifndef IMAGE_PROCESSOR
#define IMAGE_PROCESSOR
class ImageProcessor {
    public:
	virtual void preProcess(const Mat&, Mat&) = 0;
	virtual void houghLineTransform(const Mat&, Mat&) = 0;
};


#endif
