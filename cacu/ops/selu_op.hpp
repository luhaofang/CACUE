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

#ifndef SELU_OP_HPP_
#define SELU_OP_HPP_

namespace cacu {

class selu_op: public activate_base_op {

public:

	selu_op(blobs *&data) :
			activate_base_op(data, CACU_SELU) {
		_INIT_OP();
	}

	~selu_op() {

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

	void op(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_elu_cpu(s_blob_->s_data(), o_blob_->count(), _alpha, o_blob_->s_data());
		cacu_scalex_cpu(o_blob_->s_data(), o_blob_->count(), _lamda);
#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);

		cacu_elu(s_blob_->s_data(), o_blob_->count(), _alpha, o_blob_->s_data());
		cacu_scalex(o_blob_->s_data(), o_blob_->count(), _lamda);

#endif
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_selu_grad_cpu(s_blob_->s_data(), s_blob_->s_diff(), o_blob_->count(), _alpha, o_blob_->s_data(), o_blob_->s_diff());
		cacu_scalex_cpu(s_blob_->s_diff(), s_blob_->count(), _lamda);

#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);

		cacu_elu_grad(s_blob_->s_data(), s_blob_->s_diff(), o_blob_->count(), _alpha, o_blob_->s_data(), o_blob_->s_diff());
		cacu_scalex(s_blob_->s_diff(), s_blob_->count(), _lamda);
#endif
	}

	void echo() override {
		LOG_INFO("create selu op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

private:

	const float_t _alpha = 1.6732632423543772848170429916717;

	const float_t _lamda = 1.0507009873554804934193349852946;
};
}


#endif
