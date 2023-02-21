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

#include "abs_op.h"
#include "../math/math.h"

namespace cacu {

abs_op::abs_op(blobs *&data) :
		math_op(data, CACU_MATH_ABS) {
	_INIT_OP();
}

abs_op::~abs_op() {

}

void abs_op::check() {
	CHECK_EQ_OP((int )s_blobs->size(), 1, "sblobs size must == 1 vs %d",
			(int )s_blobs->size());
}

void abs_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

	cacu_abs(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
}

void abs_op::grad(blobs *&s_blobs_, blobs *&o_blobs_) {
	blob *o_blob_ = o_blobs_->asblob(0);
	blob *s_blob_ = s_blobs_->asblob(0);

	cacu_abs_grad(s_blob_->s_data(), s_blob_->s_diff(), s_blob_->count());
}

REGISTER_OP_CLASS_D(abs_op);
REGISTER_OP_MACRO(abs_op, CACU_MATH_ABS);

}

