#include "CPUImageProcessor.h"

CPUImageProcessor::CPUImageProcessor() {
    printf("Not implemented yet");
}

void CPUImageProcessor::preProcess(const Mat& frame, Mat& result) {
    Mat temp;
    cvtColor(frame, temp, COLOR_RGB2GRAY);
    
    // Apply smoothing kernel
    GaussianBlur(temp, temp, Size( 5, 5 ), 0, 0 );

    // Attempt to mask image to extract white lines
    Scalar lower(200), upper(255);

    inRange(temp, lower, upper, temp);
		
    // Apply canny edge detector
    int lowThreshold = 150;
    int ratio = 2.5;
    int kernel_size = 3;
    Canny(temp, result, lowThreshold, lowThreshold*ratio, kernel_size );

}

void CPUImageProcessor::houghLineTransform(Mat& frame, Mat& result) {
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(frame, lines, 3, 2 * CV_PI/180, 120, 0, 0 ); // runs the actual detection
	vector<Vec2f> uniqueLines;
	uniqueLines.push_back(lines.front());
	float rho_threshold = 50;
	float theta_threshold = CV_PI/6;
	for (Vec2f curr : lines) {
	    float rho = curr[0];
	    float theta = curr[1];
	    
	    int uniqueLinesSize = uniqueLines.size();
	    for (int i = 0; i < uniqueLinesSize; i++) {
		float t_rho = uniqueLines[i][0];
		float t_theta = uniqueLines[i][1];
		if ((rho > t_rho + rho_threshold || rho < t_rho - rho_threshold) && 
		    (theta > t_theta +theta_threshold || theta < t_theta - theta_threshold)) { 
		    uniqueLines.push_back(curr);
		    break;
		}
	    }

	}	
	const int SHIFT_ROWS = frame.rows * 2;
	// Draw the lines
	for (Vec2f curr : uniqueLines) {
	    float rho = curr[0], theta = curr[1];

	    Point pt1, pt2;
	    double a = cos(theta), b = sin(theta);
	    double x0 = a*rho, y0 = b*rho;
	    pt1.x = cvRound(x0 + 1250*(-b));
	    pt1.y = cvRound(y0 + 1250*(a) + SHIFT_ROWS);
	    pt2.x = cvRound(x0 - 1250*(-b));
	    pt2.y = cvRound(y0 - 1250*(a) + SHIFT_ROWS);
	    line( result, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
	}
}
