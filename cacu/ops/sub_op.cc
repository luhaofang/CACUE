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

#include "sub_op.h"
#include "../math/math.h"

namespace cacu {

	sub_op::sub_op(blobs *&data) :
		math_op(data, CACU_MATH_SUB) {
		_INIT_OP();
	}

	sub_op::~sub_op() {

	}

	void sub_op::check(){
		CHECK_EQ_OP((int)s_blobs->size(), 2, "sblobs size must == 2 vs %d",
				(int)s_blobs->size());
	}

	void sub_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		blob *_data = s_blobs_->asblob(1);

		s_blob_->_CHECK_SIZE_EQ(_data);
		cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
		cacu_saxpy(_data->s_data(), (float_t)-1, o_blob_->s_data(), _data->count());
		//cacu_print(o_blob_->s_data(),100);

	}

	void sub_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
		blob *_data = s_blobs_->asblob(1);

		s_blob_->_CHECK_SIZE_EQ(_data);
		cacu_saxpy(o_blob_->s_diff(), (float_t)1, s_blob_->s_diff(), o_blob_->count());
		cacu_saxpy(o_blob_->s_diff(), (float_t)-1, _data->s_diff(), o_blob_->count());
	}

REGISTER_OP_CLASS_D(sub_op);
REGISTER_OP_MACRO(sub_op, CACU_MATH_SUB);

}
