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

class relu_op: public operator_base {

public:

	relu_op(blob_base *&data) :
			operator_base(data, CACU_RELU) {
		check();
		initial();
		init_weights();
		echo();
	}

	~relu_op() {

	}

	virtual const void initial() override {
		if (o_blob == NULL)
			o_blob = s_blob;
		else
			o_blob->_NEED_MOTIFY();
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

		cacu_relu_cpu(o_blob_->s_data(), o_blob_->count());

#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;
		cacu_relu(o_blob_->s_data(), o_blob_->count());
#endif
	}

	virtual const void grad() override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		cacu_relu_grad_cpu(s_blob_->s_data(), o_blob_->s_diff(),
				s_blob_->count());

#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;

		cacu_relu_grad(s_blob_->s_data(),o_blob_->s_diff(), s_blob_->count());

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
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blob->channel(), s_blob->width(), s_blob->height(),
				o_blob->channel(), o_blob->width(), o_blob->height());
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
