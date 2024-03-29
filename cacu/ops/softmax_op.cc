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

#include "softmax_op.h"
#include "../math/math.h"

namespace cacu {

	softmax_op::softmax_op(blobs *&data) :
			activate_base_op(data, CACU_SOFTMAX) {
		_INIT_OP();
	}

	softmax_op::~softmax_op() {

	}

	void softmax_op::initial() {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_blobs();
			o_blobs->push_back(create_em_oblob(s_blob->num(), s_blob->channel(),
					s_blob->width(), s_blob->height(), _phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void softmax_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_softmax_cpu(s_blob_->s_data(), s_blob_->num(), s_blob_->channel(),
				s_blob_->width(), s_blob_->height(), o_blob_->s_data());

#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		cacu_softmax(s_blob_->s_data(), s_blob_->num(),s_blob_->channel(), s_blob_->width(), s_blob_->height(), o_blob_->s_data());
#endif
		//echo();
	}

	void softmax_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {
		//blob *o_blob_ = (blob*) o_blobs_->at(0);
		//blob *s_blob_ = (blob*) s_blobs_->at(0);

		//echo();

	}

	void softmax_op::echo() {
		LOG_INFO("softmax op: in(%d, %d, %d), out(%d, %d, %d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

REGISTER_OP_CLASS_D(softmax_op);
REGISTER_OP_MACRO(softmax_op, CACU_SOFTMAX);

}
