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

#include <stdio.h>
#include <string.h>

#if __PARALLELTYPE__ == __CUDA__

#include <cuda_runtime.h>

#include <cublas_v2.h>


namespace mycnn{

#define BLOCKNUM 1024
#define THREADNUM 256

//cublas log utilities
#define CUBLAS_LOG(level,status) \
		if(status!=CUBLAS_STATUS_SUCCESS) \
			{do{ fprintf(stderr,"[%s][%s %s:%d] code %d, cuda operation falled!\n",level, __TIME__, __FILE__, __LINE__,status);}while(0); exit(-1);};

#define CUBLAS_CHECK(status) CUBLAS_LOG("CUBLAS",status)


//cuda log utilities
#define CUDA_CHECK(res)   \
		if(res!=cudaSuccess) \
			{do{ fprintf(stderr,"[%s][%s %s:%d] code %d, cuda operation falled!\n","CUDA", __TIME__, __FILE__, __LINE__,res);}while(0); exit(-1);};


#define CUDA_LOG(level, format,...)   \
	if(level == "DEBUG") \
		do{ fprintf(stderr,"[CUDA_%s][%s %s:%d] %s " format "\n",level, __TIME__, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);}while(0); \
	else \
		do{ fprintf(stderr,"[CUDA_%s][%s %s:%d] " format "\n",level, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__);} while (0)

#define CUDA_DEBUG(format,...) CUDA_LOG("DEBUG",p,format,##__VA_ARGS__)
#define CUDA_INFO(format,...) CUDA_LOG("DEBUG",p,format,##__VA_ARGS__)

};


#endif
