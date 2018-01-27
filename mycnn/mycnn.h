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

#include "definition.h"
#include "config.h"

#include "math.h"

#include "framework/framework.h"
//*/

#ifndef ON
#define ON 1
#endif

#ifndef OFF
#define OFF	0
#endif

#ifndef __OPENBLAS__
#define __OPENBLAS__  0XF
#endif

#ifndef __MKL__
#define __MKL__ 0XFF
#endif

#ifndef __CUDNN__
#define __CUDNN__ 0XFFF
#endif

#ifndef __CUDA__
#define __CUDA__ 0XFFFF
#endif

#ifndef __OPENCL__
#define __OPENCL__ 0XFFFFF
#endif

#ifndef __USE_DEVICE__
#define __USE_DEVICE__  OFF
#endif

#if __USE_DEVICE__ == ON
	#ifndef __PARALLELTYPE__
		#define __PARALLELTYPE__  __CUDA__
	#endif
#else
	#ifndef __CBLASTYPE__
		#define __CBLASTYPE__   __OPENBLAS__
	#endif
#endif

#ifndef __OPENMP__
#define __OPENMP__  OFF
#endif

#ifndef __USEMBEDDING__
#define __USEMBEDDING__  OFF
#endif

//embedding size for device
#ifndef __EMBEDSIZE__
#define __EMBEDSIZE__ 1
#endif

