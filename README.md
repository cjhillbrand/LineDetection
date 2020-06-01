# LineDetection
HPC 490 Final Project: Line Detection using Hough Transform implemented on GPU and CPU code

Compile instructions for Linux:
`nvcc 'pkg-config --cflags --libs /usr/lib/pkgconfig/opencv4.pc' -L/usr/lib/ -lcuda -L/opt/cuda/lib -lcudart \*.cpp \*.cu -o LineDetection`

