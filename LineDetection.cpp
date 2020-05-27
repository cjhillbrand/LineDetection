#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <cstring>

#include "VideoProcessor.h"

int main(int argc, char* argv[]) {
    // Entry point for the Line Detection algorithm 
    // First we parse through the flags that are given
    // -c -> Designates whether or not we run the canny edge detection
    //	    on the cpu or GPU 
    // -h -> Designate whether or not we run the Hough line transform 
    //	    on the CPU or GPU
    // -f -> if this flag is set then we read in the file
    bool cannyCPU = true;
    bool houghCPU = true;
    char file[40] = "./videos/LaneDetectionTest_01.mp4";
    int verbose = 1;

    int opt;
    while ((opt = getopt(argc, argv, "c:h:v:f:")) != -1) {
	switch(opt) {
	    case 'c':
		cannyCPU = strcmp("cpu", optarg) == 0;
		break;
	    case 'h':
		houghCPU = strcmp("cpu", optarg) == 0;
		break;
	    case 'f':
		strcpy(file, optarg);
		break;
	    case 'v':
		verbose = atoi(optarg);
		break;
	    default:
		fprintf(stderr, "Usage:\n-c (cpu|gpu)\n-h (cpu|gpu)\n-v <filename>\n");
		exit(EXIT_FAILURE);
	}
    }
    VideoProcessor vp(file, cannyCPU, houghCPU);
    vp.process();
    exit(0);

}
