/*
Copyright (c) 2016, David lu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <execinfo.h>
#include <cxxabi.h>
#include <time.h>
#include "math_config.h"
#include "cuda/cuda_utils.h"
//#include "../../tools/rand_t.h"

using namespace std;
#ifdef _WIN32
#define CHECK_OP(level, format, ...) \
	do{								 \
		time_t now = time(NULL);	 \
		struct tm _now_; 			 \
		localtime_s(&_now_,&now);	 \
		fprintf(stderr,"[%s][%02d:%02d:%02d %s:%d] " format "\n", level, _now_.tm_hour,_now_.tm_min,_now_.tm_sec, __FILE__, __LINE__,##__VA_ARGS__);\
	    }while(0)
#else
#define CHECK_OP(level, format, ...) \
	do{								 \
		time_t now = time(NULL);	 \
		struct tm _now_; 			 \
		localtime_r(&now,&_now_);	 \
		fprintf(stderr,"[%s][%02d:%02d:%02d %s:%d] " format "\n", level, _now_.tm_hour,_now_.tm_min,_now_.tm_sec, __FILE__, __LINE__,##__VA_ARGS__);\
    }while(0)
#endif

#define CHECK_EQ_OP(x,y,format,...)					\
	if (x != y){							\
			void * array[32];		  \
			char ** stacktrace = backtrace_symbols(array, 1);  \
			CHECK_OP("FATAL", "%s " format,stacktrace[0],##__VA_ARGS__, x, y);	\
			free(stacktrace);				\
			exit(0);						\
			}

#define CHECK_LT_OP(x,y,format,...)					\
	if (x >= y){							\
			void * array[32];		  \
			char ** stacktrace = backtrace_symbols(array, 1);  \
			CHECK_OP("FATAL", "%s " format,stacktrace[0],##__VA_ARGS__, x, y);	\
			free(stacktrace);				\
			exit(0);						\
			}

#define CHECK_LE_OP(x,y,format,...)					\
	if (x > y){								\
			void * array[32];		  \
			char ** stacktrace = backtrace_symbols(array, 1);  \
			CHECK_OP("FATAL", "%s " format,stacktrace[0],##__VA_ARGS__, x, y);	\
			free(stacktrace);				\
			exit(0);						\
			}

#define CHECK_GT_OP(x,y,format,...)					\
	if (x <= y){							\
			void * array[32];		  \
			char ** stacktrace = backtrace_symbols(array, 1);  \
			CHECK_OP("FATAL", "%s " format,stacktrace[0],##__VA_ARGS__, x, y);	\
			free(stacktrace);				\
			exit(0);						\
			}

#define CHECK_GE_OP(x,y,format,...)					\
	if (x < y){								\
			void * array[32];		  \
			char ** stacktrace = backtrace_symbols(array, 1);  \
			CHECK_OP("FATAL", "%s " format,stacktrace[0],##__VA_ARGS__, x, y);	\
			free(stacktrace);				\
			exit(0);						\
			}




typedef enum {

	TRANS,NOTRANS

} TRANSPOSE;

typedef enum {

	BYWIDTH, BYHEIGHT

} SUM;


template<typename DTYPE>
inline void cacu_print(DTYPE *data, int length)
{

#if __PARALLELTYPE__ == __CUDA__
	cuda_print(data,length);
#else
	for(int i = 0; i < length ;++i)
		cout << data[i] << ",";
	cout << endl;
#endif
}




