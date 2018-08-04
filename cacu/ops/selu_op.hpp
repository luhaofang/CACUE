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

class selu_op: public operator_base {

public:

	selu_op(blob_base *&data) :
			operator_base(data, CACU_SELU) {
		check();
		initial();
		init_weights();
		//echo();
	}

	~selu_op() {

	}

	void initial()  {
		if (o_blob == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = s_blob;
			_rand_vect = create_em_opblob(s_blob->num(), s_blob->channel(),
					s_blob->width(), s_blob->height(), test);
#else
			o_blob = create_oblob(s_blob->num(), s_blob->channel(), s_blob->width(), s_blob->height(), _phase);
#endif
		} else {
			o_blob->resize(s_blob->num(), s_blob->channel(), s_blob->width(),
								s_blob->height());
		}
	}

	void init_weights()  {
		return;
	}

	void check()  {
		return;
	}

	void op()  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		cacu_elu_cpu(s_blob_->s_data(), o_blob_->count(), _alpha, o_blob_->s_data());
		cacu_scalex_cpu(o_blob_->s_data(), o_blob_->count(), _lamda);
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;

		cacu_elu(s_blob_->s_data(), o_blob_->count(), _alpha, o_blob_->s_data());
		cacu_scalex(o_blob_->s_data(), o_blob_->count(), _lamda);

#endif
	}

	void grad()  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		cacu_selu_grad_cpu(s_blob_->s_data(), s_blob_->s_diff(), o_blob_->count(), _alpha, o_blob_->s_data(), o_blob_->s_diff());
		cacu_scalex_cpu(s_blob_->s_diff(), s_blob_->count(), _lamda);

#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;

		cacu_elu_grad(s_blob_->s_data(), s_blob_->s_diff(), o_blob_->count(), _alpha, o_blob_->s_data(), o_blob_->s_diff());
		cacu_scalex(s_blob_->s_diff(), s_blob_->count(), _lamda);
#endif
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo()  {
		LOG_INFO("create selu op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blob->channel(), s_blob->width(), s_blob->height(),
				o_blob->channel(), o_blob->width(), o_blob->height());
	}

	inline void LOOP_INIT_DATA_()
	{
		return;
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

private:

	const float_t _alpha = 1.6732632423543772848170429916717;

	const float_t _lamda = 1.0507009873554804934193349852946;
};
}


#endif
