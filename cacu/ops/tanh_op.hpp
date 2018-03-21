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

#pragma once

namespace cacu {

class tanh_op: public operator_base {

public:

	tanh_op(blob_base *&data) :
			operator_base(data, CACU_TANH) {
		check();
		initial();
		init_weights();
		echo();
	}

	~tanh_op() {

	}

	virtual const void initial() override {
		if (o_blob == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(s_blob->num(), s_blob->channel(),
					s_blob->height(), s_blob->width(), _phase);
#else
			o_blob = create_oblob(s_blob->num(), s_blob->channel(), s_blob->width(), s_blob->height(), _phase);
#endif
		} else {
			o_blob->resize(s_blob->num(), s_blob->channel(), s_blob->width(),
					s_blob->height());
		}
	}

	virtual const void init_weights() override {
		return;
	}

	virtual const void check() override {
		return;
	}

	virtual const void op() override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		cacu_tanh_cpu(s_blob_->s_data(), o_blob_->count(), o_blob_->s_data());

#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;

		cacu_tanh(s_blob_->s_data(), o_blob_->count(), o_blob_->s_data());
#endif
	}

	virtual const void grad() override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		cacu_tanh_grad_cpu(o_blob_->s_data(), o_blob_->s_diff(),
				s_blob_->count(), s_blob_->s_diff());

#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;

		cacu_tanh_grad(o_blob_->s_data(), o_blob_->s_diff(), s_blob_->count(), s_blob_->s_diff());

#endif
	}

	virtual const void load(std::ifstream& is) override {
		return;
	}

	virtual const void save(std::ostream& os) override {
		return;
	}

	virtual const void echo() override {
		LOG_INFO("create relu op:");
		LOG_INFO(
				"channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",
				s_blob->channel(), s_blob->height(), o_blob->channel(),
				o_blob->height());
	}

	inline virtual const void LOOP_INIT_DATA_() override
	{
		return;
	}

	inline virtual const void set_phase(phase_type phase_) override {
		_phase = phase_;
	}

private:

};
}
