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

#ifndef MATH_FUNCTION_OBLAS_H_
#define MATH_FUNCTION_OBLAS_H_

#ifdef __CBLASTYPE__
#if __CBLASTYPE__ == __OPENBLAS__

#include <cblas.h>

namespace cacu{

inline void cacu_saxpy_oblas(const float *x, const float a, float *y,const int length)
{
	cblas_saxpy(length, a, x, 1, y, 1);
}

inline void cacu_saxpby_oblas(const float *x, const float a, float *y,const float b,const int length)
{
	cblas_saxpby(length, a, x, 1, b, y, 1);
}

inline void cacu_scalex_oblas(float *x,const float a,const int length)
{
	cblas_sscal(length, a, x, 1);
}

inline void cacu_sgemv_oblas(CBLAS_TRANSPOSE trans, const float *x,const int x_height,const float *y,const int x_width, const float alpha,float *z, const float beta)
{
	int m = x_height,n = x_width;
	cblas_sgemv(CblasColMajor, trans, m, n, alpha, x, m, y, 1, beta, z, 1);
}

inline void cacu_sgemm_oblas(CBLAS_TRANSPOSE transx, CBLAS_TRANSPOSE transy,const float *x,const int x_height,const int x_width,const float *y,const int y_width,const float alpha,float *z,const float beta)
{
	int m = x_height,n = y_width,k = x_width;
	int lda = (transx == CblasNoTrans) ? m : k;
	int ldb = (transy == CblasNoTrans) ? k : n;
	cblas_sgemm(CblasColMajor, transx, transy, m, n, k, alpha, x, lda, y, ldb, beta, z, m);
}

inline void cacu_copy_oblas(const float *x, const int x_length,float *y)
{
	cblas_scopy(x_length,x,1,y,1);
}

}

#endif
#endif

#endif
