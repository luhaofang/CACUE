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

#include "focal_softmax_with_loss_op.h"

#include "../math/math.h"

namespace cacu {

	focal_softmax_with_loss_op::focal_softmax_with_loss_op(blobs *&data) :
				softmax_with_loss_op(data) {
		_OP_TYPE = CACU_FOCAL_SOFTMAX_LOSS;
	}

	focal_softmax_with_loss_op::~focal_softmax_with_loss_op() {

	}

	void focal_softmax_with_loss_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {
		softmax_with_loss_op::op(s_blobs_, o_blobs_);
	}

	void focal_softmax_with_loss_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

		blob *s_blob_ = s_blobs_->asblob(0);
		blob *o_blob_ = o_blobs_->asblob(0);
		bin_blob *labels_ = s_blobs_->asbinblob(1);

		float_t alpha_ = 1;

		softmax_with_loss_op::grad(s_blobs_, o_blobs_);

#if __USEMBEDDING__ == ON
		o_blob_->s_data_cpu();
		s_blob_->s_diff_cpu();
		labels_->s_data_cpu();
#endif
		int label_;

		_train_count = 0;
		for (int i = 0; i < s_blob_->num(); ++i){

			label_ = argmax(o_blob_->p_data(i), o_blob_->length());
			if(labels_->p_data(i)[0] == label_){
				_train_count += 1;
//				s_blob_->set_pdiff(0, i);
			}
#if __USEMBEDDING__ == ON
			for(int h = 0 ; h < s_blob_->height(); ++h)
				for(int w = 0 ; w < s_blob_->width(); ++w){
					alpha_ = 1.0 - o_blob_->p_data(i)[labels_->p_data_cpu(i,false)[h * s_blob_->width() + w]
									        * s_blob_->channel_length() + h * s_blob_->width() + w];
					alpha_ = powf(alpha_, _lambda);
					for(int c = 0; c < o_blob_->channel(); ++c){
						s_blob_->p_diff_cpu(i, false)[c * s_blob_->channel_length() + h * s_blob_->width() + w] *= alpha_;
					}
				}
#else
			for(int h = 0 ; h < s_blob_->height(); ++h)
				for(int w = 0 ; w < s_blob_->width(); ++w){
					alpha_ = 1.0 - o_blob_->p_data(i)[labels_->p_data(i)[h * s_blob_->width() + w]
														* s_blob_->channel_length() + h * s_blob_->width() + w];
					alpha_ = powf(alpha_, _lambda);
					for(int c = 0; c < o_blob_->channel(); ++c){
						s_blob_->p_diff(i)[c * s_blob_->channel_length() + h * s_blob_->width() + w] *= alpha_;
					}
				}
#endif
		}
		LOG_DEBUG("conrrect predict: %d vs %d", _train_count, s_blob_->num());
#if __USEMBEDDING__ == ON
		device_copy2dev(s_blob_->s_diff(), s_blob_->s_diff_cpu(false), s_blob_->count());
#endif
	}

REGISTER_OP_CLASS_D(focal_softmax_with_loss_op);
REGISTER_OP_MACRO(focal_softmax_with_loss_op, CACU_FOCAL_SOFTMAX_LOSS);

}

