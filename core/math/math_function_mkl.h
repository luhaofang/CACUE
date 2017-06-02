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

#include "../../core/utils/data_defination.h"

#if __CBLASTYPE__ == __MKL__

#include <mkl.h>

inline void cacu_saxpy_mkl(const mycnn::float_t *x,const mycnn::float_t a, mycnn::float_t *y, int length)
{
	cblas_saxpy(length, a, x, 1, y, 1);
}

inline void cacu_saxpby_mkl(const mycnn::float_t *x,const mycnn::float_t a, mycnn::float_t *y,const mycnn::float_t b, int length)
{
	cblas_saxpby(length, a, x, 1, b, y, 1);
}

inline void cacu_scalex_mkl(mycnn::float_t *x,const mycnn::float_t a, int length)
{
	cblas_sscal(length, a, x, 1);
}

inline void cacu_sgemv_mkl(CBLAS_TRANSPOSE trans, const mycnn::float_t *x, int x_height, const mycnn::float_t *y, int x_width,const mycnn::float_t alpha,mycnn::float_t *z,const mycnn::float_t beta)
{
	int m = x_height,n = x_width;
	cblas_sgemv(CBLAS_LAYOUT::CblasColMajor, trans, m, n, alpha, x, m, y, 1, beta, z, 1);
}

inline void cacu_sgemm_mkl(CBLAS_TRANSPOSE transx, CBLAS_TRANSPOSE transy, const mycnn::float_t *x, int x_height, int x_width, const mycnn::float_t *y, int y_width, const mycnn::float_t alpha,mycnn::float_t *z,const mycnn::float_t beta)
{
	int m = x_height,n = y_width,k = x_width;
	int lda = (transx == CBLAS_TRANSPOSE::CblasNoTrans) ? m : k;
	int ldb = (transy == CBLAS_TRANSPOSE::CblasNoTrans) ? k : n;
	cblas_sgemm(CBLAS_LAYOUT::CblasColMajor, transx, transy, m, n, k, alpha, x, lda, y, ldb, beta, z, m);
}

inline void cacu_copy_mkl(const mycnn::float_t *x, int x_length,mycnn::float_t *y)
{
	cblas_scopy(x_length,x,1,y,1);
}

#endif
