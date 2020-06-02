#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "ImageProcessor.h"
#include "CPUImageProcessor.h"
#include "GPUImageProcessor.h"

#ifndef VIDEO_PROCESSOR
#define VIDEO_PROCESSOR

using namespace cv;
class VideoProcessor {
    public:
	VideoProcessor(const char*, const bool, const bool);
	~VideoProcessor();
	void process();
    private:
	VideoCapture video;
	int retrieveNextFrame(Mat&);
	void showFrame(const Mat&);
	ImageProcessor* cannyProc;
	ImageProcessor* houghProc;

	const bool cannyCPU;
	const bool houghCPU;
	const char* windowName;
	const int EXIT_CODE = -1;
	const int SUCC_CODE = 1;

};

#endif
