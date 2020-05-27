#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <iostream>

#include "ImageProcessor.h"

#ifndef VIDEO_PROCESSOR
#define VIDEO_PROCESSOR

using namespace cv;
class VideoProcessor {
    public:
	VideoProcessor(const char*);
	void process();
    private:
	VideoCapture video;
	void retrieveNextFrame(Mat&);
	void showFrame(const Mat&);
	ImageProcessor* proc;

};

#endif
