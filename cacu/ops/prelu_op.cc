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

#include "prelu_op.h"
#include "../math/math.h"

namespace cacu {

	prelu_op::prelu_op(blobs *&data) :
			activate_base_op(data, CACU_PRELU) {
		_INIT_OP();
	}

	prelu_op::~prelu_op() {

	}

	void prelu_op::initial() {
		if (o_blobs == NULL) {
			o_blobs = s_blobs;
		} else {
			o_blobs->_NEED_MOTIFY();
		}
	}

	void prelu_op::init_weights() {
		_p_slopes = create_param("pslopes", 1, s_blobs->at(0)->channel(), 1, 1, _phase);
		set_param_init_type(constant, _p_slopes, 0.25);
	}

	void prelu_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_prelu_cpu(o_blob_->s_data(), _p_slopes->s_data(), o_blob_->num(),
				o_blob_->channel(), o_blob_->channel_length());

#else
		o_blobs_ = s_blobs_;
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		//LOG_DEBUG("input");
		//cacu_print(o_blob_->s_data(), 100);
		cacu_prelu(o_blob_->s_data(), _p_slopes->s_data(), s_blob_->num(),
				s_blob_->channel(), s_blob_->channel_length());
		//LOG_DEBUG("output");
		//cacu_print(o_blob_->s_data(), 100);
#endif
	}

	void prelu_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_prelu_grad_cpu(s_blob_->s_data(), o_blob_->s_diff(),
				_p_slopes->s_data(), _p_slopes->s_diff(), o_blob_->num(),
				o_blob_->channel(), o_blob_->channel_length());

#else
		o_blobs_ = s_blobs_;
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		cacu_prelu_grad(s_blob_->s_data(), o_blob_->s_diff(), _p_slopes->s_data(), _p_slopes->s_diff(), o_blob_->num(),
				o_blob_->channel(), o_blob_->channel_length());

#endif
	}

	void prelu_op::load(std::ifstream& is) {
		_p_slopes->load(is);
	}

	void prelu_op::save(std::ostream& os) {
		_p_slopes->serializa(os);
	}

	void prelu_op::echo() {
		LOG_INFO("prelu op: in(%d, %d, %d), out(%d, %d, %d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

REGISTER_OP_CLASS_D(prelu_op);
REGISTER_OP_MACRO(prelu_op, CACU_PRELU);


}
