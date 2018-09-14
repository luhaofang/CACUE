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

#ifndef ELU_OP_HPP_
#define ELU_OP_HPP_

namespace cacu {

class elu_op: public activate_base_op {

public:

	elu_op(blobs *&data, op_args *&args_) :
		activate_base_op(data, args_, CACU_ELU) {
		_INIT_OP();
	}

	~elu_op() {

	}

	void initial() override {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
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

	void check() override {
		if(_o_args == NULL)
			LOG_FATAL("elu op args cannot equal to NULL!");
		//negative_slope > 0
		CHECK_GT_OP(_o_args->at(0), 0, "alpha must > 0 vs %d",
				_o_args->at(0));
	}

	void op(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_elu_cpu(s_blob_->s_data(), o_blob_->count(), _o_args->at(0), o_blob_->s_data());
		cacu_scalex_cpu(o_blob_->s_data(), o_blob_->count(), _lamda);
#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);

		cacu_elu(s_blob_->s_data(), o_blob_->count(), _o_args->at(0), o_blob_->s_data());

#endif
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_elu_grad_cpu(s_blob_->s_data(), s_blob_->s_diff(), o_blob_->count(), _o_args->at(0), o_blob_->s_data(), o_blob_->s_diff());

#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);

		cacu_elu_grad(s_blob_->s_data(), s_blob_->s_diff(), o_blob_->count(), _o_args->at(0), o_blob_->s_data(), o_blob_->s_diff());
#endif
	}

	void echo() override {
		LOG_INFO("create elu op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

};
}


#endif
