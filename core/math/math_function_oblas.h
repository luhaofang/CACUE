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

#include <cblas.h>


inline void cacu_saxpy_oblas(float_t *x, float_t a, float_t *y, int length)
{
	cblas_saxpy(length, a, x, 1, y, 1);
}

inline void cacu_saxpby_oblas(float_t *x, float_t a, float_t *y, float_t b, int length)
{
	cblas_saxpby(length, a, x, 1, b, y, 1);
}

inline void cacu_scalex_oblas(float_t *x, float_t a, int length)
{
	cblas_sscal(length, a, x, 1);
}

inline void cacu_sgemv_oblas(CBLAS_TRANSPOSE trans, float_t *x, int x_height, float_t *y, int x_width, float_t *z)
{
	cblas_sgemv(CblasRowMajor, trans, x_height, x_width, (float_t)1, x, x_width, y, 1, (float_t)0, z, 1);
}

inline void cacu_sgemm_oblas(CBLAS_TRANSPOSE transx, CBLAS_TRANSPOSE transy, float_t *x, int x_height, int x_width, float_t *y, int y_width, float_t *z)
{
	int m = x_height,n = y_width,k = x_width;
	int lda = (transx == CblasNoTrans) ? k : m;
	int ldb = (transy == CblasNoTrans) ? n : k;
	cblas_sgemm(CblasRowMajor, transx, transy, m, n, k, 1, x, lda, y, ldb, 0, z, n);
}

inline void cacu_transpose_oblas(float_t *x, int x_width, int x_height)
{

}

inline void cacu_copy_oblas(float_t *x, int x_length,float_t *y)
{
	cblas_scopy(x_length,x,1,y,1);
}
