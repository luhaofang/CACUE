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

#ifndef CONFIG_H_
#define CONFIG_H_

namespace cacu {

/*
 * Blas definiation & parallal dnn defination
 *
 * This is used for global configuration for CACUE use different calculation.
 * __OPENBLAS__ : for using openblas for blas computation, and basic math calculation.
 *  __MKL__ 	: link with Intel MKL for math calculation.
 * 	__CUDNN__ 	: create the basic dnn operator calculation by Nvidia CuDNN library.
 *	__CUDA__	: blas creator with introduce the cublas math methods to accelerate computing.
 * 	__OPENCL__  : using opencl for outer device computation.
 *
 */
//openblas
#define __OPENBLAS__  0XA1
//mkl
#define __MKL__ 0XA2
//cudnn
#define __CUDNN__ 0XA3
//cuda & cublas
#define __CUDA__ 0XA4
//opencl
#define __OPENCL__ 0XA5


/***********************************/
/*       calculation method	       */
/***********************************/

#define __DYNAMIC_GRAPH__  0XB1
#define __STATIC_GRAPH__  0XB2

#ifndef __OPERATOR__TYPE__
#define __OPERATOR__TYPE__ __STATIC_GRAPH__
#endif


/***********************************/
/*        user config part	       */
/***********************************/

#ifndef __USE_DEVICE__
#define __USE_DEVICE__  OFF
#endif

#ifndef __PARALLELTYPE__
#define __PARALLELTYPE__  __OPENBLAS__
#endif

#ifndef __CBLASTYPE__
#if(__PARALLELTYPE__ == __CUDA__ || __PARALLELTYPE__ == __CUDNN__ || __PARALLELTYPE__ == __OPENCL__)
#define __CBLASTYPE__ __OPENBLAS__
#else
#define __CBLASTYPE__ __PARALLELTYPE__
#endif
#endif

#ifndef __USEMBEDDING__
#define __USEMBEDDING__  OFF
#endif

//embedding size for device
#ifndef __EMBEDSIZE__
#define __EMBEDSIZE__ 1
#endif

}


#endif
