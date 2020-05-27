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
    printf("Process function not implemented yet");
}

int VideoProcessor::retrieveNextFrame(Mat& frame) {
    return video.read(frame) ? SUCC_CODE : EXIT_CODE;
}

void VideoProcessor::showFrame(const Mat& frame) {
    imshow(windowName, frame);
    // May have to enter the "waitKey(0)" function call here... not sure.
}
