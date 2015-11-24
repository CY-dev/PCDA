CPPFLAGS = -ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread 
CPP = g++

all: mex_PCDA_openmp
mex_PCDA_openmp: mex_PCDA.cpp cdfit.cpp common.h
	mex CXX=$(CPP) CXXFLAGS="-fopenmp $(CPPFLAGS)" -largeArrayDims -lgomp mex_PCDA.cpp cdfit.cpp
