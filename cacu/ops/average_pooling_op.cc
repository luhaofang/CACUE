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

#include "average_pooling_op.h"
#include "../math/math.h"

namespace cacu {

average_pooling_op::average_pooling_op(blobs *&data, data_args *&args_) :
		pooling_base_op(data, args_, CACU_AVERAGE_POOLING) {
	_INIT_OP();

}

average_pooling_op::~average_pooling_op() {

}

void average_pooling_op::initial() {

	int output_w = (s_blobs->at(0)->width() - _args->kernel_size())
			/ _args->stride() + 1;
	int output_h = (s_blobs->at(0)->height() - _args->kernel_size())
			/ _args->stride() + 1;
	int pad = abs(
			s_blobs->at(0)->width() - (output_w - 1) * _args->stride()
					- _args->kernel_size());
	if (pad != 0)
		output_w += 1;
	pad = abs(
			s_blobs->at(0)->height() - (output_h - 1) * _args->stride()
					- _args->kernel_size());
	_args->at(3) = pad;
	if (pad != 0)
		output_h += 1;

	if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
		o_blobs = create_em_oblobs();
		o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), output_w,
						output_h, _phase));
#else
		o_blobs = create_oblobs();
		o_blobs->push_back(
				create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
						output_w, output_h, _phase));
#endif
	} else {
		o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
				output_w, output_h);
	}
#if __USE_CUDNN__ == ON
	blob *s_blob_ = s_blobs->asblob(0);
	if (!s_blob_->body()->check_body(_NONE_BLOB)) {
		set_pooling_desc(_pooling_desc,
				CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
				_args->kernel_size(), _args->kernel_size(), 0, 0,
				_args->stride(), _args->stride());
	}
#endif
}

void average_pooling_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {
#if __USEMBEDDING__ == ON
	em_blob *o_blob_ = (em_blob*) o_blob;
	em_blob *s_blob_ = (em_blob*) s_blob;
	for (int i = 0; i < s_blob_->num(); ++i) {
		cacu_average_pooling(s_blob_->p_data_d(i), _args->kernel_size(),
				_args->stride(), s_blob_->width(), s_blob_->height(),
				o_blob_->width(), o_blob_->height(), s_blob_->channel(),
				o_blob_->p_data_d(i));
		o_blob_->_sync(i);
	}
#else
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

#if __USE_CUDNN__ == OFF
	for(int i = 0; i < s_blob_->num(); ++i)
	cacu_average_pooling(s_blob_->p_data(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->height(), o_blob_->width(), o_blob_->height(), s_blob_->channel(), o_blob_->p_data(i));
#else

	cudnn_pooling_forward(_pooling_handle, s_blob_->tensor_desc(),
			s_blob_->s_data(), _pooling_desc, o_blob_->tensor_desc(),
			o_blob_->s_data());
#endif
#endif
}

void average_pooling_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {
#if __USEMBEDDING__ == ON
	em_blob *o_blob_ = (em_blob*) o_blob;
	em_blob *s_blob_ = (em_blob*) s_blob;
	for (int i = 0; i < s_blob_->num(); ++i) {
		cacu_average_pooling_grad(o_blob_->p_diff_d(i),
				_args->kernel_size(), _args->stride(), s_blob_->width(),
				s_blob_->height(), o_blob_->width(), o_blob_->height(),
				s_blob_->channel(), s_blob_->p_diff_d(i));
		s_blob_->_sync(i);
	}
#else
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

#if __USE_CUDNN__ == OFF
	for(int i = 0; i < s_blob_->num(); ++i)
	cacu_average_pooling_grad(o_blob_->p_diff(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->height(), o_blob_->width(), o_blob_->height(), s_blob_->channel(), s_blob_->p_diff(i));
#else
	cudnn_pooling_backward(_pooling_handle, s_blob_->tensor_desc(),
			s_blob_->s_data(), s_blob_->tensor_desc(), s_blob_->s_diff(),
			_pooling_desc, o_blob_->tensor_desc(), o_blob_->s_data(),
			o_blob_->tensor_desc(), o_blob_->s_diff());
#endif
#endif
}

void average_pooling_op::echo() {
	LOG_INFO(
			"average pooling op: in(%d, %d, %d), out(%d, %d, %d), kernel/stride/pad([%d, %d], %d, %d)",
			s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
			s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
			o_blobs->at(0)->width(), o_blobs->at(0)->height(),
			_args->kernel_size(), _args->kernel_size(), _args->stride(),
			_args->pad());
}

REGISTER_OP_CLASS_D_DP(average_pooling_op);
REGISTER_OP_MACRO(average_pooling_op, CACU_AVERAGE_POOLING);

}
