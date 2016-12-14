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

namespace mycnn{


	template<typename DTYPE>
	extern "C" void cacu_sumbysize_gpu(SUM SUMTYPE ,DTYPE *x, int length, DTYPE *y, int width);


	template<typename DTYPE>
	extern "C" void cacu_cxsize_gpu(DTYPE *x, int length, DTYPE *a, int size,DTYPE *y);


	template<typename DTYPE>
	extern "C" void cacu_sxsize(DTYPE *x, int length, DTYPE a, DTYPE *y);


	template<typename DTYPE>
	extern "C" void cacu_cdxsize(DTYPE *x, int length, DTYPE *a, int size, DTYPE *y);


	template<typename DTYPE>
	extern "C" void cacu_sdxsize(DTYPE *x, int length, DTYPE a, DTYPE *y);


	template<typename DTYPE>
	extern "C" void cacu_ssxpy(DTYPE *x, DTYPE a, int size, DTYPE *y, DTYPE b, int length, DTYPE *z);


	template<typename DTYPE>
	extern "C" void cacu_sqr(DTYPE *x, int length, DTYPE *y);

	template<typename DTYPE>
	extern "C" void cacu_root(DTYPE *x, int length, DTYPE *y);


	template<typename DTYPE>
	extern "C" void cacu_stdbychannel(DTYPE *x, int length, DTYPE *y, DTYPE epsilon);


};
