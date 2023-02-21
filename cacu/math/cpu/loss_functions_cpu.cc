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

#include "loss_functions_cpu.h"

#include <algorithm>
#include <cmath>

namespace cacu {


void cacu_cross_entropy_multi_cpu(float_t *x, const int num, const int channel, const int width, const int height,
		const int *label_, float_t *loss_){

	float *xp;

	int c_length = width * height;
	int length = channel * c_length;
	int index, n, h, w;

	#if __OPENMP__ == ON
	#pragma omp parallel for default(shared) private(n,xp)
	#endif
	for (n = 0; n < num; ++n) {
		for (h= 0; h< height;++h)
			for(w = 0; w < width; ++w)
			{
				index = h * width + w;
				xp = x + n * length + index;
				loss_[0] -= (label_[index + n * c_length] >= 0) ? logf(max(xp[label_[index + n * c_length]*c_length], float_t(_MIN_FLT_))) : 0;
			}
	}

}


void cacu_cross_entropy_multi_grad_cpu(float_t *x, const int num, const int channel, const int width, const int height, const int *label_, float_t *xg){

	float *xp;

	int c_length = width * height;
	int length = channel * c_length;
	int index, n, h, w;

	#if __OPENMP__ == ON
	#pragma omp parallel for default(shared) private(n,xp)
	#endif
	for (n = 0; n < num; ++n) {
		for (h= 0; h< height;++h)
			for(w = 0; w < width; ++w) {
				index = h * width + w;
				xp = x + n * length + index;
				xg[n * length + index + label_[index + n * c_length] * c_length] =
						float_t(-1.0) / max(xp[label_[index + n * c_length]*c_length], float_t(_MIN_FLT_));
			}
	}

}


}
