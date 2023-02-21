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

#include "concentrate_op.h"

#include "../math/math_functions.h"


namespace cacu {

	concentrate_op::concentrate_op(blobs *&data, op_args *&args_) :
			operator_base(data, args_, CACU_CONCENTRATE) {
		_INIT_OP();
	}

	concentrate_op::~concentrate_op() {

	}

	void concentrate_op::initial() {

		int concentrate_size = (int)_o_args->at(0);

		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), _phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num() / concentrate_size, s_blobs->at(0)->channel() * concentrate_size,
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num() / concentrate_size, s_blobs->at(0)->channel() * concentrate_size,
					s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void concentrate_op::init_weights() {
		return;
	}

	void concentrate_op::check() {
		if(_o_args == NULL)
			LOG_FATAL("concentrate op args cannot equal to NULL!");
		int concentrate_size = _o_args->at(0);
		//split count > 0
		CHECK_EQ_OP(s_blobs->at(0)->num() % concentrate_size, 0, "s_blob num must integrated times of concentrate_size = 0 vs %d",
				s_blobs->at(0)->num() % concentrate_size);
	}

	void concentrate_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);

		for (int i = 0; i < o_blob_->num(); ++i) {
			for (unsigned int j = 0; j < (s_blobs)->size(); ++j) {
				em_blob *s_blob_ = (em_blob *) s_blobs->at(j);
				cacu_saxpy(s_blob_->p_data_d(i), (float_t) 1,
						o_blob_->p_data_d(i), o_blob_->length());
			}
			o_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());

#endif
	}

	void concentrate_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);

		for (unsigned int j = 0; j < (s_blobs)->size(); ++j) {
			em_blob *s_blob_ = (em_blob *) s_blobs->at(j);
			cacu_copy_cpu(o_blob_->s_diff(), o_blob_->count(),
					s_blob_->s_diff());
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		cacu_copy(o_blob_->s_diff(), s_blob_->count(), s_blob_->s_diff());
#endif
	}

	void concentrate_op::load(std::ifstream& is) {
		return;
	}

	void concentrate_op::save(std::ostream& os) {
		return;
	}

	void concentrate_op::echo() {
		LOG_INFO("combination op: in(%d, %d, %d), out(%d, %d, %d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}


REGISTER_OP_CLASS_D_OP(concentrate_op);
REGISTER_OP_MACRO(concentrate_op, CACU_CONCENTRATE);

}
