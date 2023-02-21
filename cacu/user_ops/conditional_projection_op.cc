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

#include "conditional_projection_op.h"

#include "../math/math.h"

namespace cacu {

	conditional_projection_op::conditional_projection_op(blobs *&data) :
			operator_base(data, CACU_CONDITIONAL_PROJECTION) {
		_INIT_OP();
	}

	conditional_projection_op::~conditional_projection_op() {

	}

	void conditional_projection_op::initial() {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs = create_em_oblob(s_blobs->at(0)->num(), _args->output_channel(), 1,
					1, _phase);
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), 1, s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), 1, s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void conditional_projection_op::init_weights() {
		return;
	}

	void conditional_projection_op::check() {
		CHECK_EQ_OP((int)s_blobs->size(), 2, "input data size must be equal to 2 vs %d",
				(int)s_blobs->size());
		//input data batch size equal to each other
		CHECK_EQ_OP(s_blobs->at(0)->num(), s_blobs->at(1)->num(), "input data must be equal %d vs %d",
				s_blobs->at(0)->num(), s_blobs->at(1)->num());
	}

	void conditional_projection_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_sgemm(TRANS, NOTRANS, _w->s_data(), _w->num(), _w->length(),
					s_blob_->p_data_d(i), 1, 1, o_blob_->p_data_d(i), 0);
			//bias added
			if (_is_use_bias)
				cacu_saxpby(_bias->s_data(), (float_t) (1),
						o_blob_->p_data_d(i), (float_t) (1), _bias->count());
			//cacu_ssxpy(_bias->s_data(),(float_t)(1),_bias->count(), o_blob_->p_data_d(i),(float_t)1,o_blob_->length(),o_blob_->p_data_d(i));
			o_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		bin_blob *label = s_blobs_->asbinblob(1);

		vec_i _target(label->count());
#if __USE_DEVICE__ == ON
		cuda_copy2host(&_target[0], label->s_data(), label->count());
#else
		for(int i = 0 ; i < s_blob_->num(); ++i)
			_target[i] = label->s_data()[i];
#endif
		for(int i = 0; i < s_blob_->num(); ++i)
			cacu_copy(s_blob_->p_data(i) + _target[i] * s_blob_->channel_length(), s_blob_->channel_length(), o_blob_->p_data(i));
#endif
	}

	void conditional_projection_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		for (int i = 0; i < s_blob_->num(); ++i) {
			//gradient propagation
			cacu_sgemm(NOTRANS, NOTRANS, _w->s_data(), _w->length(), _w->num(),
					o_blob_->p_diff_d(i), 1, 1, s_blob_->p_diff_d(i), 0);
			//weights gradient
			cacu_sgemm(NOTRANS, TRANS, s_blob_->p_data_d(i), s_blob_->length(),
					1, o_blob_->p_diff_d(i), o_blob_->length(), 1, _w->s_diff(),
					1);
			if (_is_use_bias)
				//bias gradient
				cacu_sumbysize(BYHEIGHT, o_blob_->p_diff_d(i),
						o_blob_->length(), 1, _bias->s_diff(), 1,
						_bias->count());
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		bin_blob *label = s_blobs_->asbinblob(1);

		vec_i _target(label->count());
#if __USE_DEVICE__ == ON
		cuda_copy2host(&_target[0], label->s_data(), label->count());
#else
		for(int i = 0 ; i < s_blob_->num(); ++i)
			_target[i] = label->s_data()[i];
#endif
//		cacu_print(o_blob_->s_diff(), o_blob_->count());

		for(int i = 0; i < s_blob_->num(); ++i)
			cacu_copy(o_blob_->p_diff(i), s_blob_->channel_length(), s_blob_->p_diff(i) + _target[i] * s_blob_->channel_length());

//		cacu_print(s_blob_->s_diff(), s_blob_->count());
#endif
	}

	void conditional_projection_op::load(std::ifstream& is) {
		return;
	}

	void conditional_projection_op::save(std::ostream& os) {
		return;
	}

	void conditional_projection_op::echo()
	{
		LOG_INFO("conditional projection op: in(%d, %d, %d), out(%d, %d, %d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

REGISTER_OP_CLASS_D(conditional_projection_op);
REGISTER_OP_MACRO(conditional_projection_op, CACU_CONDITIONAL_PROJECTION);


}
