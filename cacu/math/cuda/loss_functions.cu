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

#include "../../config.h"


#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__

#include "../../definition.h"
#include "../../tensor/cuda/cuda_log.h"

namespace cacu{

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
__global__ void _k_CACU_CROSS_ENTROPY_CUDA(float_t *x, int num, int length, unsigned int *label_, float_t *loss_) {

	int tid = threadIdx.x;

	extern __shared__ float_t shared_data[];

	float_t *xp;

	shared_data[tid] = 0;

	for (int i = tid; i < num; i+=THREADNUM)
	{
		xp = x + i * length;
		shared_data[tid] -= log(max(xp[label_[i]], float_t(_MIN_FLT_)));
	}

	__syncthreads();

	int acc_length = THREADNUM / 2;
	while(acc_length > 0){
		if(tid < acc_length)
			shared_data[tid] += shared_data[tid + acc_length];
		acc_length /= 2;
		__syncthreads();
	}

	if(tid == 0)
		loss_[0] += shared_data[0];
}


extern "C" void cacu_cross_entropy_cuda(float_t *x, int num, int length, unsigned int *label_, float_t *loss_){

	_k_CACU_CROSS_ENTROPY_CUDA<<<1, THREADNUM, THREADNUM * sizeof(float_t)>>>(x, num, length, label_,loss_);
	CUDA_CHECK(cudaThreadSynchronize());
}

}

#endif
#endif
