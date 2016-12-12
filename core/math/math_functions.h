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

#include "math_function_openmp.h"
#include "math_function_oblas.h"

namespace mycnn{

template<typename DTYPE>
void cacu_saxpy(DTYPE *x, DTYPE a, DTYPE *y,int length)
{

#if __PARALLELTYPE__ == __OPENMP__
	cacu_saxpby_opm(x, a, y, length);
#elif __PARALLELTYPE__ == __OPENBLAS__
	cacu_saxpy_oblas(x, a, y, length);
#elif __PARALLELTYPE__ == __GPU__
	LOG_INFO("Haven't finished yet!");
#endif
}

template<typename DTYPE>
void cacu_caxpy(DTYPE *x, DTYPE *a, DTYPE *y, int length)
{

#if __PARALLELTYPE__ == __OPENMP__
	cacu_saxpby_opm(x, a, y, length);
#elif __PARALLELTYPE__ == __OPENBLAS__
	cacu_caxpy_oblas(x, a, y,length);
#elif __PARALLELTYPE__ == __GPU__
	LOG_INFO("Haven't finished yet!");
#endif
}

template<typename DTYPE>
void cacu_saxpby(DTYPE *x, DTYPE a, DTYPE *y, DTYPE b, int length)
{

#if __PARALLELTYPE__ == __OPENMP__
	cacu_saxpby_opm(x, a, y, b, length);
#elif __PARALLELTYPE__ == __OPENBLAS__
	cacu_saxpby_oblas(x, a, y, b, length);
#elif __PARALLELTYPE__ == __GPU__
	LOG_INFO("Haven't finished yet!");
#endif

}

template<typename DTYPE>
void cacu_caxpby(DTYPE *x, DTYPE *a, DTYPE *y, DTYPE *b, int length)
{

#if __PARALLELTYPE__ == __OPENMP__
	cacu_caxpby_opm(x, a, y, b, length);
#elif __PARALLELTYPE__ == __OPENBLAS__
	cacu_caxpby_oblas(x, a, y, b, length);
#elif __PARALLELTYPE__ == __GPU__
	LOG_INFO("Haven't finished yet!");
#endif

}

template<typename DTYPE>
void cacu_sgemv(TRANSPOSE trans,DTYPE *x, int x_height, DTYPE *y, int x_width, DTYPE *z)
{
	
#if __PARALLELTYPE__ == __OPENMP__
	cacu_sgemv_opm(x, x_height, y, x_width, z);
#elif __PARALLELTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx = (trans == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemv_oblas(transx, x, x_height, y, x_width, z);
#elif __PARALLELTYPE__ == __GPU__
	LOG_INFO("Haven't finished yet!");
#endif
}

template<typename DTYPE>
void cacu_sgemm(TRANSPOSE transx_, TRANSPOSE transy_, DTYPE *x, int x_height, int x_width, DTYPE *y, int y_height, DTYPE *z)
{

#if __PARALLELTYPE__ == __OPENMP__
	LOG_INFO("Haven't finished yet!");
#elif __PARALLELTYPE__ == __OPENBLAS__
	CBLAS_TRANSPOSE transx = (transx_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	CBLAS_TRANSPOSE transy = (transy_ == TRANS) ? CBLAS_TRANSPOSE::CblasTrans : CBLAS_TRANSPOSE::CblasNoTrans;
	cacu_sgemm_oblas(transx, transy, x, x_height, x_width, y, y_height, z);
#elif __PARALLELTYPE__ == __GPU__
	LOG_INFO("Haven't finished yet!");
#endif
}


};