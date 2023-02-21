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

#include "scalex_op.h"
#include "../math/math.h"

namespace cacu {

	scalex_op::scalex_op(blobs *&data, op_args *&args_) :
		math_op(data, args_, CACU_MATH_SCALEX) {
		_INIT_OP();
	}

	scalex_op::~scalex_op() {
	}

	void scalex_op::initial() {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->height(), s_blobs->at(0)->width(), _phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height());
		}
	}

	void scalex_op::check() {
		if(_o_args == NULL)
			LOG_FATAL("scalex op args cannot equal to NULL!");

		_scaler = _o_args->at(0);
		//parameter a, b
		CHECK_EQ_OP((int)_o_args->size(), 1, "scalex parameter must == 1 vs %d",
				(int)_o_args->size());
		CHECK_EQ_OP((int)s_blobs->size(), 1, "sblobs size must == 1 vs %d",
				(int)s_blobs->size());
	}

	void scalex_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

		blob *s_blob_ = s_blobs_->asblob(0);
		blob *o_blob_ = o_blobs_->asblob(0);

		o_blob_->copy2data(s_blob_);
		cacu_scalex(o_blob_->s_data(),  o_blob_->count(), _scaler);
	}

	void scalex_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

		blob *s_blob_ = s_blobs_->asblob(0);
		blob *o_blob_ = o_blobs_->asblob(0);

		s_blob_->copy2diff(o_blob_);
		cacu_scalex(s_blob_->s_diff(), s_blob_->count(), _scaler);
	}

	void scalex_op::echo(){
		LOG_INFO("scalex op: in(%d, %d, %d), out(%d, %d, %d), scaler: %f",
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
					o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height(), _scaler);
	}

REGISTER_OP_CLASS_D_OP(scalex_op);
REGISTER_OP_MACRO(scalex_op, CACU_MATH_SCALEX);

}
