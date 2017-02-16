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


	/**
	 * @cacu_sumbysize
	 * sum by size:
	 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
	 * sum by width y is (length/ width) height dim, sum by height y is width dim.
	 */
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

	/**
	 * @cacu_cxsize
	 * math y[i] = a[j]*x[i] :
	 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding scalar, j = i / (length / size).
	 */
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

	/**
	 * @cacu_cdxsize
	 * math y[i] = x[i] / a[j] :
	 * x is a length dim array list, a is a size dim array list, a[j] is the corresponding denominator, j = i / (length / size).
	 */
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

	/**
	 * @cacu_sdxsize
	 * math y[i] = x[i] / a :
	 * x is a length dim array list, a is the corresponding denominator.
	 */
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

	/**
	 * @cacu_ssxpy
	 * math z[i] = a * x[j] + b * y[i] :
	 * y is a length dim array list, x is a size dim array list, x[j] is the corresponding scalar, j = i / (length / size).
	 * a & b are corresponding scalars for x, y
	 */
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

	/**
	 * @cacu_sqr
	 * math y[i] = x[i]^2 :
	 */
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

	/**
	 * @cacu_root
	 * math y[i] = sqrt(x[i]) :
	 */
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

	/**
	 * @cacu_stdbychannel
	 * math std[i] = sqrt(varience[i] + epsilon) :
	 */
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

	/**
	 * @cacu_ssx
	 * math y[i] *= x[i] :
	 * scale by element wise.
	 */
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
