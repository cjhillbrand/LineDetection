#include "VideoProcessor.h"

VideoProcessor::VideoProcessor(const char* file, const bool cannyCPU, const bool houghCPU) : 
	cannyCPU(cannyCPU), houghCPU(houghCPU), windowName(file), video(VideoCapture()) {
    
    video.open(file);
    
    if (!video.isOpened()) {
	fprintf(stderr, "Unable to open video %s... Exiting\n", file);
	exit(EXIT_FAILURE);
    }
    
    if (cannyCPU) {
        cannyProc = new CPUImageProcessor();
    }
    else {
        cannyProc = new GPUImageProcessor();
    }

    if (houghCPU) {
        houghProc = new CPUImageProcessor();
    }
    else {
        houghProc = new GPUImageProcessor();
    }

    // Create the window that we will display the image to.
    namedWindow(windowName, WINDOW_AUTOSIZE);
}

VideoProcessor::~VideoProcessor() {
    delete cannyProc;
    delete houghProc;

}

void VideoProcessor::process() {
    Mat frame;
    while (retrieveNextFrame(frame) != EXIT_CODE) {
	Mat result(frame);
	Mat cutFrame = frame(Rect(0, frame.rows - frame.rows/3, frame.cols, frame.rows/3));
    Mat preHough(cutFrame.rows, cutFrame.cols, CV_8UC1);
	cannyProc -> preProcess(cutFrame, preHough);
	imshow("PRE HOUGH", preHough);
	houghProc -> houghLineTransform(preHough, frame);
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
