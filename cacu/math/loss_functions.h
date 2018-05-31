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

#ifndef LOSS_FUNCTIONS_H_
#define LOSS_FUNCTIONS_H_

#include "math_definition.h"
#include "../config.h"
#include "../definition.h"

#include "cuda/loss_functions_cuda.h"
#include "cpu/loss_functions_cpu.h"

namespace cacu {

/**
 * @cacu_cross_entropy
 * math x[i] = max(0,x[i]) :
 * for loss use cross entropy functions.
 */
inline void cacu_cross_entropy(const float_t *x, const int num, const int length,
		const int *label_, float_t *loss_) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_cross_entropy_cuda(x, num, length, label_, loss_);
#endif
#else
	cacu_cross_entropy_cpu(x, num, length, label_,loss_);
#endif
}

/**
 * @cacu_cross_entropy
 * math x[i] = max(0,x[i]) :
 * for loss use cross entropy functions.
 */
inline void cacu_cross_entropy_multi(const float_t *x, const int num, const int channel, const int width, const int height,
		const int *label_, float_t *loss_) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_cross_entropy_multi_cuda(x, num, channel, width, height, label_, loss_);
#endif
#else
	cacu_cross_entropy_multi_cpu(x, num, channel, width, height, label_, loss_);
#endif
}

/**
 * @cacu_multi_label_trans
 * transform the softmax label to multi sigmoid labels
 */
inline void cacu_multi_label_trans(const int num, const int output_num,
		const int *label_, int *trans_labels_) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_multi_label_trans_cuda(num, output_num, label_, trans_labels_);
#endif
#else

	for (int n = 0; n < num; ++n)
	{
		for(int i = 0; i < output_num; ++i)
		trans_labels_[n * output_num + i] = 0;
		trans_labels_[n * output_num + label_[n]] = 1;
	}
#endif
}

}


#endif
