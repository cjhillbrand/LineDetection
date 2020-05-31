#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;

#ifndef IMAGE_PROCESSOR
#define IMAGE_PROCESSOR
class ImageProcessor {
    public:
	virtual void preProcess(const Mat&, Mat&) = 0;
	virtual void houghLineTransform(const Mat&, Mat&) = 0;
};


#endif
