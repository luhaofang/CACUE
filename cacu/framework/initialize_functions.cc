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


#include "initialize_functions.h"
#include "../math/math_definition.h"
//#include "../tensor/utils/device_data_utils.h"
#include "../math/utils/rand_t.h"

#include "blob.h"

namespace cacu {

void gaussian_init(blob *&blob_, float_t mean, float_t std){
	vec_t w(blob_->count());
	for (int i = 0; i < blob_->count(); ++i){
		w[i] = gaussrand(std, mean);
	}
	blob_->copy2data(w);
	vec_t().swap(w);
}

void msra_init(blob *&blob_, float_t scaling){
	vec_t w(blob_->count());
	float_t d_value = sqrt((float_t) 2.0 / (blob_->length()));
	for (int i = 0; i < blob_->count(); ++i)
		w[i] = gaussrand(d_value) * scaling;
	blob_->copy2data(w);
	vec_t().swap(w);
}

void uniform_init(blob *&blob_, float_t min, float_t max){
	CHECK_GT_OP(max, min, "max value must >= min value : %f vs %f!", max, min);
	vec_t w(blob_->count());
	for (int i = 0; i < blob_->count(); ++i)
		w[i] = urand(min, max);
	blob_->copy2data(w);
	vec_t().swap(w);
}

void xavier_init(blob *&blob_, float_t scaling){
	vec_t w(blob_->count());
	float_t d_value = sqrt((float_t) 6.0 / (blob_->length() + blob_->count() / blob_->channel()));
	for (int i = 0; i < blob_->count(); ++i)
		w[i] = urand(-d_value, d_value) * scaling;
	blob_->copy2data(w);
	vec_t().swap(w);
}

void gaussian_init(float_t *w_, const int length, float_t mean, float_t std){
	vec_t w(length);
	for (int i = 0; i < length; ++i){
		w[i] = gaussrand(std, mean);
	}
#if __USE_DEVICE__ == ON
	device_copy2dev(w_, &w[0], length);
#else
	cacu_copy_cpu(&w[0], length, w_);
#endif
	vec_t().swap(w);
}

}
