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

#include "injector_op.h"
#include "../math/math.h"

namespace cacu {

	injector_op::injector_op(blobs *&data) :
			operator_base(data, CACU_INJECTOR) {
		_INIT_OP();
	}

	injector_op::~injector_op() {

	}

	void injector_op::initial() {
		if (o_blobs == NULL) {
			o_blobs = s_blobs;
			_mean->resize(1, o_blobs->at(0)->channel(), 1, 1);
			_mask->resize(1, o_blobs->at(0)->channel(), o_blobs->at(0)->height(),
					o_blobs->at(0)->width());
		}
	}

	void injector_op::init_weights() {
		return;
	}

	void injector_op::check() {
		return;
	}

	void injector_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_sumbysize(BYWIDTH, s_blob_->p_data_d(i), s_blob_->length(), 1,
					_mean->s_data(), 0, s_blob_->height() * s_blob_->width());
			cacu_scalex(_mean->s_data(), _mean->count(),
					(float_t) (1.0 / (s_blob_->height() * s_blob_->width())));
			cacu_ssxpy(_mean->s_data(), (float_t) (-1), _mean->count(),
					s_blob_->p_data_d(i), (float_t) 1, s_blob_->count(),
					o_blob_->p_data_d(i));
			cacu_relu(o_blob_ - p_data_d(i), o_blob_->length());
			mask_vector(o_blob_->p_data_d(i), o_blob_->length(),
					_mask->s_data());
			cacu_ssxpy(_mean->s_data(), (float_t) (1), _mean->count(),
					s_blob_->p_data_d(i), (float_t) 1, s_blob_->length(),
					o_blob_->p_data_d(i));
			cacu_ssx(_mask->s_data(), _mask->count(), s_blob_->p_data_d(i));
			o_blob_->_sync(i);
		}

#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		for(int i = 0; i < s_blob_->num(); ++i)
		{
			cacu_sumbysize(BYWIDTH,s_blob_->p_data(i),s_blob_->length(),1,_mean->s_data(),0,s_blob_->height()*s_blob_->width());
			cacu_scalex(_mean->s_data(),_mean->count(),((float_t)1.0/(s_blob_->height()*s_blob_->width())));
			cacu_ssxpy(_mean->s_data(),(float_t)(-1),_mean->count(),s_blob_->p_data(i),(float_t)1,s_blob_->length(),o_blob_->p_data(i));
			cacu_relu(o_blob_->p_data(i),o_blob_->length());
			mask_vector(o_blob_->p_data(i),o_blob_->length(),_mask->s_data());
			cacu_ssxpy(_mean->s_data(),(float_t)(1),_mean->count(),s_blob_->p_data(i),(float_t)1,s_blob_->length(),o_blob_->p_data(i));
			cacu_ssx(_mask->s_data(),_mask->count(), s_blob_->p_data(i));
		}
#endif
	}

	void injector_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {
		return;
	}

	void injector_op::load(std::ifstream& is) {
		return;
	}

	void injector_op::save(std::ostream& os) {
		return;
	}

	void injector_op::echo()
	{
		LOG_INFO("create injector op:");
		LOG_INFO(
				"channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",
				s_blobs->at(0)->channel(), s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
				o_blobs->at(0)->height());
	}

REGISTER_OP_CLASS_D(injector_op);
REGISTER_OP_MACRO(injector_op, CACU_INJECTOR);

}
