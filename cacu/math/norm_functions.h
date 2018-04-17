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

#ifndef NORM_FUNCTIONS_H_
#define NORM_FUNCTIONS_H_

#include "math_definition.h"
#include "../config.h"
#include "../definition.h"

#include "cuda/norm_functions_cuda.h"
#include "cpu/norm_functions_cpu.h"

namespace cacu {

/**
 * @cacu_cross_entropy
 * math x[i] = max(0,x[i]) :
 * for loss use cross entropy functions.
 */
inline void cacu_norm_l1(float_t *x, int num, int length,
		const unsigned int *label_, float_t *loss_) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

#endif
#else

#endif
}

/**
 * @cacu_cross_entropy
 * math x[i] = max(0,x[i]) :
 * for loss use cross entropy functions.
 */
inline void cacu_norm_l2(float_t *x, int num, int length,
		const unsigned int *label_, float_t *loss_) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

#endif
#else

#endif
}

}


#endif
