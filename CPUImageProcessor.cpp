#include "VideoProcessor.h"

VideoProcessor::VideoProcessor(const char* file, const bool cannyCPU, const bool houghCPU) : 
	cannyCPU(cannyCPU), houghCPU(houghCPU), windowName(file), video(VideoCapture()) {
    
    video.open(file);
    
    if (!video.isOpened()) {
	fprintf(stderr, "Unable to open video %s... Exiting\n", file);
	exit(EXIT_FAILURE);
    }
    
    if (cannyCPU || houghCPU)  // Going to have to change the logic bc what if one is
    // true and another is false?
	proc = new CPUImageProcessor();
    else
	proc = new GPUImageProcessor();

    // Create the window that we will display the image to.
    namedWindow(windowName, WINDOW_AUTOSIZE);
}

VideoProcessor::~VideoProcessor() {
    delete proc;
}

void VideoProcessor::process() {
    Mat frame;
    Mat preHough;

    while (retrieveNextFrame(frame) != EXIT_CODE) {
	    Mat result(frame);
   
        const int cols = frame.cols;
        const int rows = frame.rows;

        Mat cutFrame = frame(Rect(0, frame.rows - frame.rows / 3, frame.cols, frame.rows / 3));

	    proc -> preProcess(cutFrame, preHough);
	    imshow("PRE HOUGH", preHough);
	    proc -> houghLineTransform(preHough, frame);
	    showFrame(result);
    }
}

int VideoProcessor::retrieveNextFrame(Mat& frame) {
    return video.read(frame);
}

void VideoProcessor::showFrame(const Mat& frame) {
    imshow(windowName, frame);
    waitKey(0);
    // May have to enter the "waitKey(0)" function call here... not sure.
}
