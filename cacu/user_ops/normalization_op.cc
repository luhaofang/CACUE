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

#include "normalization_op.h"
#include "../math/math.h"


namespace cacu {

	normalization_op::normalization_op(blobs *&data) :
			operator_base(data, CACU_NORM) {
		_INIT_OP();
	}

	normalization_op::~normalization_op() {

	}

	void normalization_op::initial() {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_blobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->height(), s_blobs->at(0)->width(), _phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->height(), s_blobs->at(0)->width(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->height(),
					s_blobs->at(0)->width());
		}
	}

	void normalization_op::init_weights() {
		return;
	}

	void normalization_op::check() {

	}

	void normalization_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {
#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		em_blob *temp_ = (em_blob*) _temp;

		for (int i = 0; i < s_blob_->num(); ++i) {
			//cacu_max_pooling(s_blob_->p_data_d(i), _args->kernel_size(), _args->stride(), s_blob_->width(), o_blob_->width(), s_blob_->channel(), o_blob_->p_data_d(i), index_->p_data_d(i));
			o_blob_->_sync(i);
			temp_->_sync(i);
		}
#else
//		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		switch(_NORMALIZER)
		{
			case norm_type::norm_l1:
				for(int i = 0; i < s_blob_->num(); ++i) {
					cacu_l1_normalization(s_blob_->p_data(i), s_blob_->length(), _epsilon);
				}
				break;
			case norm_type::norm_l2:
				for(int i = 0; i < s_blob_->num(); ++i) {
					cacu_l2_normalization(s_blob_->p_data(i), s_blob_->length(), _epsilon);
				}
				break;
			default:
				break;
		}
#endif
	}

	void normalization_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		em_bin_blob *temp_ = (em_bin_blob*) _temp;
		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_max_pooling_grad(o_blob_->p_diff_d(i), _args->kernel_size(),
					_args->stride(), s_blob_->width(), o_blob_->width(),
					s_blob_->channel(), s_blob_->p_diff_d(i),
					index_->p_data_d(i));
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		switch(_NORMALIZER)
		{
			case norm_l1:
				for(int i = 0; i < s_blob_->num(); ++i) {
					cacu_l1_normalization_grad(s_blob_->p_data(i), s_blob_->length(), _epsilon, o_blob_->p_diff(i), s_blob_->p_diff(i));
				}
				break;
			case norm_l2:
				for(int i = 0; i < s_blob_->num(); ++i) {
					cacu_l2_normalization_grad(s_blob_->p_data(i), s_blob_->length(), _epsilon, o_blob_->p_diff(i), s_blob_->p_diff(i));
				}
				break;
			default:
				break;
		}
#endif
	}

	void normalization_op::load(std::ifstream& is) {
		return;
	}

	void normalization_op::save(std::ostream& os) {
		return;
	}

	void normalization_op::echo() {
		LOG_INFO("normalization op: in(%d, %d, %d), out(%d, %d, %d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}


REGISTER_OP_CLASS_D(normalization_op);
REGISTER_OP_MACRO(normalization_op, CACU_NORM);

}
