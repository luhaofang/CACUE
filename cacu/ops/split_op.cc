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

#include "split_op.h"
#include "../math/math.h"

namespace cacu {

	split_op::split_op(blobs *&data, op_args *&args_) :
			operator_base(data, args_, CACU_SPLIT) {
		_INIT_OP();
	}

	split_op::~split_op() {

	}

	void split_op::initial() {
		_split_count = (int)_o_args->at(0);
		if (o_blobs == NULL) {
			o_blobs = create_oblobs();
			for (int i = 0; i < _split_count; ++i) {
#if __USEMBEDDING__ == ON
				o_blobs->push_back(
						cacu_allocator::create_em_blob(s_blob->num(),
								s_blob->channel(), s_blob->width(),
								s_blob->height(), 0, _phase));
#else
				o_blobs->push_back(cacu_allocator::create_blob(s_blobs->at(0)->num(),s_blobs->at(0)->channel(),
						s_blobs->at(0)->width(),s_blobs->at(0)->height(),0,_phase));
#endif
			}
		} else {
			for (int i = 0; i < _split_count; ++i) {
				o_blobs->at(i)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
						s_blobs->at(0)->width(), s_blobs->at(0)->height());
			}
		}
	}

	void split_op::init_weights() {
		return;
	}

	void split_op::check() {
		if(_o_args == NULL)
			LOG_FATAL("split op args cannot equal to NULL!");
		int split_count = _o_args->at(0);
		//split count > 0
		CHECK_GT_OP(split_count, 0, "output_channel must > 0 vs %d",
				split_count);
	}

	void split_op::op(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		for (int j = 0; j < (o_blobs)->size(); ++j) {
			em_blob *o_blob_ = (em_blob *) o_blobs->at(j);
			cacu_copy_cpu(s_blob_->s_data(), s_blob_->count(),
					o_blob_->s_data());
		}
#else
		blob *s_blob_ = s_blobs_->asblob(0);

		for (unsigned int j = 0; j < (o_blobs_)->size(); ++j) {
			blob *o_blob_ = o_blobs_->asblob(j);
			cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
		}
#endif
	}

	void split_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		for (int i = 0; i < s_blob_->num(); ++i) {
			for (int j = 0; j < (o_blobs)->size(); ++j) {
				em_blob *o_blob_ = (em_blob *) o_blobs->at(j);
				cacu_saxpy(o_blob_->p_diff_d(i), (float_t) 1,
						s_blob_->p_diff_d(i), o_blob_->length());
			}
			s_blob_->_sync(i);
		}
#else
		blob *s_blob_ = s_blobs_->asblob(0);

		for (unsigned int j = 0; j < (o_blobs_)->size(); ++j) {
			blob *o_blob_ = o_blobs_->asblob(j);
			cacu_saxpy(o_blob_->s_diff(),(float_t)1,s_blob_->s_diff(),o_blob_->count());
		}
#endif
	}

	void split_op::load(std::ifstream& is) {
		return;
	}

	void split_op::save(std::ostream& os) {
		return;
	}

	void split_op::echo() {
		LOG_INFO("split op: in(%d, %d, %d), out(%d, %d, %d), split_num(%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(),
				o_blobs->at(0)->height(), _split_count);
	}


REGISTER_OP_CLASS_D_OP(split_op);
REGISTER_OP_MACRO(split_op, CACU_SPLIT);

}
