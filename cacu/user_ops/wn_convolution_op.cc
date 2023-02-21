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

#include "wn_convolution_op.h"

#include "../math/math.h"

namespace cacu {

	//output_channel, kernel_size, stride, pad, input_dim, channel
	wn_convolution_op::wn_convolution_op(blobs *&data, data_args *&args_) :
			convolution_op(data, args_) {
		_OP_TYPE = CACU_WN_CONVOLUTION;
		convolution_op::set_is_use_bias(false);
	}

	wn_convolution_op::~wn_convolution_op() {

		if(_w_original != NULL)
			delete _w_original;
	}


	void wn_convolution_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {
		norm_weight(get_weight(0));
		convolution_op::op(s_blobs_, o_blobs_);
	}

	void wn_convolution_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {
		convolution_op::grad(s_blobs_, o_blobs_);
		norm_recover(get_weight(0));
	}

	void wn_convolution_op::echo()
	{
		LOG_INFO(
				"wn convolution op: in(%d, %d, %d), out(%d, %d, %d), kernel/stride/pad([%d, %d], %d, %d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height(),
				_args->kernel_size(), _args->kernel_size(), _args->stride(), _args->pad());
	}

	void wn_convolution_op::norm_weight(weight *&w_) {
		if(_w_original == NULL){
			LOG_INFO("Initalize the parameters for spectral normalization!");
			_w_original = new blob(w_ ->num(),w_->channel(), w_->width(), w_->height(), 0, test);
		}
		cacu_copy(w_->s_data(), w_->count(), _w_original->s_data());
		cacu_l2_normalization(w_->s_data(), w_->count(), 1e-8);
	}

	void wn_convolution_op::norm_recover(weight *&w_) {
		cacu_copy(_w_original->s_data(), w_->count(), w_->s_data());
		cacu_l2_normalization_grad(w_->s_data(), w_->count(), 1e-8, w_->s_diff(), w_->s_diff());
	}


REGISTER_OP_CLASS_D_DP(wn_convolution_op);
REGISTER_OP_MACRO(wn_convolution_op, CACU_WN_CONVOLUTION);

}
