#pragma once

#ifdef __linux__

#include <stdlib.h>
#include <time.h>
#ifndef CLOCK_MONOTONIC_RAW
#include <sys/time.h>

typedef struct timeval timestamp;
inline timestamp getTimestamp(void){
	struct timeval t;
	gettimeofday(&t, NULL);
	return t;
}
inline float getElapsedtime(timestamp t){
	struct timeval tn;
	gettimeofday(&tn, NULL);
	return (tn.tv_sec - t.tv_sec) * 1000.0f + (tn.tv_usec - t.tv_usec) / 1000.0f;
}
#else
typedef struct timespec timestamp;
inline timestamp getTimestamp(void){
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC_RAW, &t);
	return t;
}
inline double getElapsedtime(timestamp t){
	struct timespec tn;
	clock_gettime(CLOCK_MONOTONIC_RAW, &tn);
	return (double)(tn.tv_sec - t.tv_sec) * 1000.0 + (tn.tv_nsec - t.tv_nsec) / 1000000.0;
}
#endif

#else

#include <time.h>

typedef clock_t timestamp;
inline timestamp getTimestamp(void){
	return clock();
}
inline double getElapsedtime(timestamp t){
	return ((double)clock()-t) / CLOCKS_PER_SEC * 1000.0;
}

#endif

