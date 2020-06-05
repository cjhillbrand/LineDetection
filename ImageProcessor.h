#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
using namespace cv;

#ifndef IMAGE_PROCESSOR
#define IMAGE_PROCESSOR
class ImageProcessor {
    public:
	virtual void preProcess(const Mat&, Mat&) = 0;
	virtual void houghLineTransform(Mat&, Mat&) = 0;
	ImageProcessor(std::string filename) {
		fout.open(filename);
	}

	protected:
		std::chrono::steady_clock::time_point start;
		std::chrono::steady_clock::time_point stop;
		double duration = 0;
		int fd;
		std::ofstream fout;
	
	void Start() {
		start = std::chrono::high_resolution_clock::now();
	}

	void Stop() {
		stop = std::chrono::high_resolution_clock::now();
		duration += getDuration();
	}

	double getDuration() {
		return (double)std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000000;
	}

	void printTime(std::string title) {
		std::cout << "Execution time for " << title << ": " << std::right 
						<< std::setw(15) << std::setprecision(6) << getDuration() << " seconds" << std::endl;
	}

	void printDuration() {
		std::cout << "Total Frame Execution Time   :" << std::right << std::setw(16) 
						<< std::setprecision(6) << duration << " seconds" << std::endl;
	}

	void resetDuration() {
		duration = 0;
	}

	void printToFile(int isGPU, int function) {
		fout << isGPU << ' ' << function << ' ' << duration << std::endl;
	}
};


#endif
