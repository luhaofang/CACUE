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




template<typename DTYPE>
void cacu_saxpby_omp(DTYPE *x, DTYPE a, DTYPE *y, DTYPE b, int length)
{
	#pragma omp parallel for
	for (int i = 0; i < length; ++i)
	{
		x[i] = x[i] * a + y[i] * b;
	}
}

template<typename DTYPE>
void cacu_caxpby_omp(DTYPE *x, DTYPE *a, DTYPE *y, DTYPE *b, int length)
{
	#pragma omp parallel for
	for (int i = 0; i < length; ++i)
	{
		x[i] = x[i] * a[i] + y[i] * b[i];
	}
}

template<typename DTYPE>
void cacu_sgemv_omp(DTYPE *x, int x_height, DTYPE *y, int x_width, DTYPE *z)
{
	#pragma omp parallel for
	for (int i = 0; i < x_height; ++i)
	{
		for (int j = 0; j < x_width; ++j)
			z[i] += x[i*x_width + j] * y[j];
	}
}

template<typename DTYPE>
void cacu_sgemm_omp(TRANSPOSE transx, TRANSPOSE transy, DTYPE *x, int x_height, int x_width, DTYPE *y, int y_width, DTYPE *z)
{

}

template<typename DTYPE>
void cacu_transpose_omp(DTYPE *x, int width, int height)
{
		
}

template<typename DTYPE>
void cacu_copy_omp(DTYPE *x, int x_length, DTYPE *y)
{
#pragma omp parallel for
	for (int i = 0; i < x_length; ++i)
	{	
		y[i] = x[i];
	}
}

