#include "CPUImageProcessor.h"

CPUImageProcessor::CPUImageProcessor() {
    printf("Not implemented yet");
}

void CPUImageProcessor::preProcess(const Mat& frame, Mat& result) {
    // Apply smoothing kernel
    Mat smoothImg;
    GaussianBlur( frame, smoothImg, Size( 5, 5 ), 0, 0 );
    
    Mat hsvImg;
    cvtColor(smoothImg, hsvImg, COLOR_RGB2HSV);

    // Attempt to mask image to extract white lines
    Scalar lower(0, 0, 180), upper(180, 25, 255);

    Mat maskedImg;
    inRange(hsvImg, lower, upper, maskedImg);

    // Change to black and white
    //Mat greyImg;
    //cvtColor(maskedImg, greyImg, COLOR_HSV2GRAY);
    //Mat hsvChannels[3];
    //split(maskedImg, hsvChannels);
    //Mat greyImg(hsvChannels[2]);
    //imshow("BW", greyImg);
    //waitKey(0);

		
    // Apply canny edge detector
    int lowThreshold = 125;
    int ratio = 2.5;
    int kernel_size = 3;
    Canny(maskedImg, result, lowThreshold, lowThreshold*ratio, kernel_size );

}

void CPUImageProcessor::houghLineTransform(const Mat& frame, Mat& result) {
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(frame, lines, 3, 2 * CV_PI/180, 120, 0, 0 ); // runs the actual detection
	vector<Vec2f> uniqueLines;
	uniqueLines.push_back(lines.front());
	float rho_threshold = 50;
	float theta_threshold = CV_PI/6;
	for (Vec2f curr : lines) {
	    float rho = curr[0];
	    float theta = curr[1];
	    
	    for (Vec2f t : uniqueLines) {
		float t_rho = t[0];
		float t_theta = t[1];
		if ((rho > t_rho + rho_threshold || rho < t_rho - rho_threshold) && 
		    (theta > t_theta +theta_threshold || theta < t_theta - theta_threshold)) 
		    uniqueLines.push_back(curr);
	    }

	}	
	// Draw the lines
    	for(Vec2f curr : uniqueLines) {
	    float rho = curr[0], theta = curr[1];
	    Point pt1, pt2;
	    double a = cos(theta), b = sin(theta);
	    double x0 = a*rho, y0 = b*rho;
	    pt1.x = cvRound(x0 + 1250*(-b));
	    pt1.y = cvRound(y0 + 1250*(a));
	    pt2.x = cvRound(x0 - 1250*(-b));
	    pt2.y = cvRound(y0 - 1250*(a));
	    line( result, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
    	}
}
