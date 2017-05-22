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


#include "../utils/data_defination.h"


namespace mycnn{

	/**
	 * @cacu_ram_copy
	 * copy ram data
	 * math y = x:
	 * length: the input data's size
	 */
	inline void cacu_copy_cpu(float_t *x, int length, float_t *y)
	{
		memcpy(y,x,length*sizeof(float_t));
	}


	inline void cacu_saxpy_cpu(mycnn::float_t *x, mycnn::float_t a, mycnn::float_t *y, int length)
	{
#if __CBLASTYPE__ == __OPENBLAS__
		cacu_saxpy_oblas(x, a, y, length);
#elif __CBLASTYPE__ == __MKL__
		cacu_saxpy_mkl(x, a, y, length);
#endif
	}

	inline void cacu_scalex_cpu(mycnn::float_t *x, mycnn::float_t a, int length)
	{
#if __CBLASTYPE__ == __OPENBLAS__
		cacu_scalex_oblas(x, a, length);
#elif __CBLASTYPE__ == __MKL__
		cacu_scalex_mkl(x, a, length);
#endif
	}

	template<typename DTYPE>
	inline void cacu_memset(DTYPE *x, DTYPE value, int length)
	{
		for(int i = 0; i < length ; ++i)
		{
			x[i] = value;
		}
	}

};
