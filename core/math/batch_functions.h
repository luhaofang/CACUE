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

#include "cuda/batch_functions_cuda.h"

#include <math.h>


namespace mycnn{


	template<typename DTYPE>
	inline void cacu_sumbysize(SUM SUMTYPE ,DTYPE *x, int length, DTYPE *y, int width)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_sumbysize_gpu(SUMTYPE,x,length,y,width);
#else
		int height = length / width;
		DTYPE *xp;
		if (BYWIDTH == SUMTYPE){
			for (int b = 0; b < height; ++b){
				xp = x + b*width;
				for (int i = 0; i < width; ++i)
					y[b] += xp[i];
			}
		}
		else if (BYHEIGHT == SUMTYPE){
			for (int b = 0; b < height; ++b){
				xp = x + b*width;
				for (int i = 0; i < width; ++i)
					y[i] += xp[i];
			}
		}
#endif
	}

	template<typename DTYPE>
	inline void cacu_cxsize(DTYPE *x, int length, DTYPE *a, int size,DTYPE *y)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_cxsize_gpu(x, length, a, size,y);
#else
		int block_size = length / size;
		DTYPE *xp;
		DTYPE *yp;
		for (int b = 0; b < size; ++b)
		{
			xp = x + b*block_size;
			yp = y + b*block_size;
			for (int j = 0; j < block_size; ++j)
				yp[j] = xp[j] * a[b];
		}
#endif
	}

	template<typename DTYPE>
	inline void cacu_sxsize(DTYPE *x, int length, DTYPE a, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_sxsize_gpu(x, length, a, y);
#else
		for (int j = 0; j < length; ++j)
			y[j] = x[j] * a;
#endif
	}

	template<typename DTYPE>
	inline void cacu_cdxsize(DTYPE *x, int length, DTYPE *a, int size, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_cdxsize_gpu(x, length, a, size, y);
#else
		int block_size = length / size;
		DTYPE *xp;
		DTYPE *yp;
		for (int b = 0; b < size; ++b)
		{
			xp = x + b*block_size;
			yp = y + b*block_size;
			for (int j = 0; j < block_size; ++j)
				yp[j] = xp[j] / a[b];
		}
#endif
	}

	template<typename DTYPE>
	inline void cacu_sdxsize(DTYPE *x, int length, DTYPE a, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_sdxsize_gpu(x,length,a,y);
#else
		for (int j = 0; j < length; ++j)
			y[j] = x[j] / a;
#endif
	}

	template<typename DTYPE>
	inline void cacu_ssxpy(DTYPE *x, DTYPE a, int size, DTYPE *y, DTYPE b, int length, DTYPE *z)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_ssxpy_gpu(x, a, size, y, b, length, z);
#else
		int block_size = length / size;
		DTYPE *yp,*zp;
		for (int i = 0; i < size; ++i){
			yp = y + i*block_size;
			zp = z + i*block_size;
			for (int j = 0; j < block_size; ++j)
				zp[j] = a*x[i] + b*yp[j];
		}
#endif
	}


	template<typename DTYPE>
	inline void cacu_sqr(DTYPE *x, int length, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_sqr_gpu(x,length,y);
#else
		for (int j = 0; j < length; ++j)
			y[j] = x[j] * x[j];
#endif
	}

	template<typename DTYPE>
	inline void cacu_root(DTYPE *x, int length, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_root_gpu(x,length,y);
#else
		for (int j = 0; j < length; ++j)
			y[j] = sqrtf((float_t)x[j]);
#endif
	}

	template<typename DTYPE>
	inline void cacu_stdbychannel(DTYPE *varience, int length, DTYPE *std, DTYPE epsilon)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_stdbychannel_gpu(varience,length,std,epsilon);
#else
		for (int j = 0; j < length; ++j)
			std[j] = (float_t)pow(varience[j] + epsilon, 0.5);
#endif
	}

	template<typename DTYPE>
	inline void cacu_ssx(DTYPE *x, int length, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_ssx_gpu(x, length, y);
#else
		for(int i = 0 ; i < length ; ++i)
			y[i] *= x[i];
#endif
	}

};
