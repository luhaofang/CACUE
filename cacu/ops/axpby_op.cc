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

#include "axpby_op.h"
#include "../math/math.h"

namespace cacu {


	axpby_op::axpby_op(blobs *&data, op_args *&args_) :
			math_op(data, args_, CACU_MATH_AXPBY) {
			_INIT_OP();
		}

	axpby_op::~axpby_op() {
		}

	void axpby_op::check() {
		if(_o_args == NULL)
			LOG_FATAL("axpby op args cannot equal to NULL!");
		//parameter a, b
		CHECK_EQ_OP((int)_o_args->size(), 2, "axpby parameter must == 2 vs %d",
				(int)_o_args->size());
		CHECK_EQ_OP((int)s_blobs->size(), 2, "sblobs size must == 2 vs %d",
				(int)s_blobs->size());
	}

	void axpby_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		blob *_data = s_blobs_->asblob(1);

		s_blob_->_CHECK_SIZE_EQ(_data);
		cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
		cacu_saxpby(_data->s_data(), _o_args->at(0), o_blob_->s_data(), _o_args->at(1), _data->count());
	}

	void axpby_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		blob *_data = s_blobs_->asblob(1);

		s_blob_->_CHECK_SIZE_EQ(_data);
		cacu_copy(o_blob_->s_diff(), o_blob_->count(), s_blob_->s_diff());
		cacu_scalex(s_blob_->s_diff(), o_blob_->count(), _o_args->at(0));
		cacu_copy(o_blob_->s_diff(), o_blob_->count(), _data->s_diff());
		cacu_scalex(_data->s_diff(), o_blob_->count(), _o_args->at(1));
	}

REGISTER_OP_CLASS_D_OP(axpby_op);
REGISTER_OP_MACRO(axpby_op, CACU_MATH_AXPBY);

}
