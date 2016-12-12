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


template<typename DTYPE>
void cacu_saxpy_oblas(DTYPE *x, DTYPE a, DTYPE *y, int length)
{
	cblas_saxpy(length, a, x, 1, y, 1);
}

template<typename DTYPE>
void cacu_caxpy_oblas(DTYPE *x, DTYPE *a, DTYPE *y, int length)
{
	cblas_caxpy(length, a, x, 1, y, 1);
}

template<typename DTYPE>
void cacu_saxpby_oblas(DTYPE *x, DTYPE a, DTYPE *y, DTYPE b, int length)
{
	cblas_saxpby(length, a, x, 1, b, y, 1);
}

template<typename DTYPE>
void cacu_caxpby_oblas(DTYPE *x, DTYPE *a, DTYPE *y, DTYPE *b, int length)
{
	cblas_caxpby(length, a, x, 1, b, y, 1);
}

template<typename DTYPE>
void cacu_sgemv_oblas(CBLAS_TRANSPOSE trans, DTYPE *x, int x_height, DTYPE *y, int x_width, DTYPE *z)
{
	cblas_sgemv(CblasRowMajor, trans, x_height, x_width, (DTYPE)1, x, x_width, y, 1, (DTYPE)0, z, 1);
}

template<typename DTYPE>
void cacu_sgemm_oblas(CBLAS_TRANSPOSE transx, CBLAS_TRANSPOSE transy, DTYPE *x, int x_height, int x_width, DTYPE *y, int y_width, DTYPE *z)
{
	int lda = (transx == CblasNoTrans) ? x_width : x_height;
	int ldb = (transy == CblasNoTrans) ? y_width : x_width;
	cblas_sgemm(CblasRowMajor, transx, transy, x_height, y_width, x_width, 1, x, lda, y, ldb, 0, z, y_width);
}

template<typename DTYPE>
void cacu_transpose_oblas(DTYPE *x, int x_width, int x_height)
{

}
