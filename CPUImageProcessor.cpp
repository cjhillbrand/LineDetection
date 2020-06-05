#include "CPUImageProcessor.h"

CPUImageProcessor::CPUImageProcessor(std::string filename) : ImageProcessor(filename) {}

void CPUImageProcessor::preProcess(const Mat& frame, Mat& result) {
	Mat temp;
	const int cols = frame.cols, rows = frame.rows;
    cvtColor(frame, temp, COLOR_RGB2GRAY);

    // Apply smoothing kernel
    GaussianBlur(temp, temp, Size( 5, 5 ), 0, 0 );

    // Attempt to mask image to extract white lines
    Scalar lower(210), upper(255);

    inRange(temp, lower, upper, temp);
		
    // Apply canny edge detector
    int lowThreshold = 150;
    int ratio = 2.5;
    int kernel_size = 3;
    Canny(temp, result, lowThreshold, (double)lowThreshold*ratio, kernel_size );
}

void CPUImageProcessor::houghLineTransform(Mat& frame, Mat& result) {
	vector<Vec4i> lines; // will hold the results of the detection
	HoughLinesP(frame, lines, 1, CV_PI/180, 40, 10, 250 ); // runs the actual detection
	float rho_threshold = 50;
	float theta_threshold = CV_PI/6;
	const int SHIFT_ROWS = frame.rows * 3;
	const int SHIFT_COLS = frame.cols / 2;

	// Draw the lines
	for (Vec4i curr : lines) {
		// First function y = 3/5 x + 0, y = frame.rows / 4, y = -3/5 x + intercept : intercept = 3/5 * frame.cols
		//if (y0 < frame.rows - 3 / 5 * x0 && y0 < frame.rows - frame.rows/4 && y0 < frame.rows - (-3/5 * x0 + 3/5 * frame.cols) ) {
		line(result, Point(curr[0] + SHIFT_COLS, curr[1] + SHIFT_ROWS), Point(curr[2] + SHIFT_COLS, curr[3] + SHIFT_ROWS), Scalar(0, 0, 255), 3, LINE_AA);
	}
}
