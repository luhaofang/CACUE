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

#ifndef PRELU_OP_HPP_
#define PRELU_OP_HPP_

namespace cacu {

class prelu_op: public activate_base_op {

public:

	prelu_op(blobs *&data) :
			activate_base_op(data, CACU_PRELU) {
		_INIT_OP();
	}

	~prelu_op() {

	}

	void initial() override {
		if (o_blobs == NULL) {
			o_blobs = s_blobs;
		} else {
			o_blobs->_NEED_MOTIFY();
		}
	}

	void init_weights() override {
		_p_slopes = create_param("pslopes", 1, s_blobs->at(0)->channel(), 1, 1, _phase);
		set_param_init_type(constant, _p_slopes, 0.25);
	}

	void op(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_prelu_cpu(o_blob_->s_data(), _p_slopes->s_data(), o_blob_->num(),
				o_blob_->channel(), o_blob_->channel_length());

#else
		o_blobs_ = s_blobs_;
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);
		//LOG_DEBUG("input");
		//cacu_print(o_blob_->s_data(), 100);
		cacu_prelu(o_blob_->s_data(), _p_slopes->s_data(), s_blob_->num(),
				s_blob_->channel(), s_blob_->channel_length());
		//LOG_DEBUG("output");
		//cacu_print(o_blob_->s_data(), 100);
#endif
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_prelu_grad_cpu(s_blob_->s_data(), o_blob_->s_diff(),
				_p_slopes->s_data(), _p_slopes->s_diff(), o_blob_->num(),
				o_blob_->channel(), o_blob_->channel_length());

#else
		o_blobs_ = s_blobs_;
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);

		cacu_prelu_grad(s_blob_->s_data(), o_blob_->s_diff(), _p_slopes->s_data(), _p_slopes->s_diff(), o_blob_->num(),
				o_blob_->channel(), o_blob_->channel_length());

#endif
	}

	void load(std::ifstream& is) override {
		_p_slopes->load(is);
	}

	void save(std::ostream& os) override {
		_p_slopes->serializa(os);
	}

	void echo() override {
		LOG_INFO("create relu op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

private:

	weight * _p_slopes = NULL;

};
}


#endif
