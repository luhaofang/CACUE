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

#include "mean_op.h"
#include "../math/math.h"

namespace cacu {

	mean_op::mean_op(blobs *&data) :
		math_op(data, CACU_MATH_MEAN) {
		_INIT_OP();
	}

	mean_op::~mean_op() {
	}


	void mean_op::initial() {
		if (o_blobs == NULL){
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(1,1,1,1, _phase));
			_multiplier = create_opblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
					(float_t) (1), test);
			_multiplier->set_variable(false);
		}
		else{
			o_blobs->at(0)->resize(1,1,1,1);
			_multiplier->resize(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height());
			_multiplier->set_data(1.0);
		}
	}

	void mean_op::check() {
		CHECK_EQ_OP((int)s_blobs->size(), 1, "sblobs size must == 1 vs %d",
				(int)s_blobs->size());
	}

	void mean_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		cacu_sgemv(NOTRANS, s_blob_->s_data(), 1, _multiplier->s_data(), s_blob_->count(), (float_t)(1.0f/ s_blob_->count()), o_blob_->s_data(), (float_t)0);
	}

	void mean_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		s_blob_->set_diff((float_t)(1.0f/ s_blob_->count()));
		cacu_cxsize(s_blob_->s_diff(), s_blob_->count(), o_blob_->s_diff(), 1, s_blob_->s_diff());
	}

REGISTER_OP_CLASS_D(mean_op);
REGISTER_OP_MACRO(mean_op, CACU_MATH_MEAN);

}
