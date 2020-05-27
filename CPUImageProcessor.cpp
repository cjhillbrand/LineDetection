#include "CPUImageProcessor.h"

CPUImageProcessor::CPUImageProcessor() {
    printf("Not implemented yet");
}

void CPUImageProcessor::preProcess(const Mat& frame, Mat& result) {
    // Change to black and white
    Mat greyImg;
    cvtColor(frame, greyImg, COLOR_RGB2GRAY);
    
    // Apply smoothing kernel
    Mat smoothImg;
		GaussianBlur( greyImg, smoothImg, Size( 5, 5 ), 0, 0 );
		
    // Apply canny edge detector
    int lowThreshold = 0;
    int ratio = 3;
    int kernel_size = 3;
    Canny(smoothImg, result, lowThreshold, lowThreshold*ratio, kernel_size );

}

void CPUImageProcessor::houghLineTransform(const Mat& frame, Mat& result) {
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(frame, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
	// Draw the lines
    	for( size_t i = 0; i < lines.size(); i++ ) {
	    float rho = lines[i][0], theta = lines[i][1];
	    Point pt1, pt2;
	    double a = cos(theta), b = sin(theta);
	    double x0 = a*rho, y0 = b*rho;
	    pt1.x = cvRound(x0 + 1000*(-b));
	    pt1.y = cvRound(y0 + 1000*(a));
	    pt2.x = cvRound(x0 - 1000*(-b));
	    pt2.y = cvRound(y0 - 1000*(a));
	    line( result, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
    	}
}
