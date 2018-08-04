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

#ifndef SOFTMAX_OP_HPP_
#define SOFTMAX_OP_HPP_

namespace cacu {

class softmax_op: public operator_base {

public:

	softmax_op(blob_base *&data) :
			operator_base(data, CACU_SOFTMAX) {
		check();
		initial();
		init_weights();
		//echo();

	}

	~softmax_op() {

	}

	void initial()  {
		if (o_blob == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(s_blob->num(), s_blob->channel(),
					s_blob->width(), s_blob->height(), _phase);
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

		cacu_softmax_cpu(s_blob_->s_data(), s_blob_->num(), s_blob_->channel(),
				s_blob_->width(), s_blob_->height(), o_blob_->s_data());

#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;
		cacu_softmax(s_blob_->s_data(), s_blob_->num(),s_blob_->channel(), s_blob_->width(), s_blob_->height(), o_blob_->s_data());
#endif
		//echo();
	}

	void grad()  {
		blob *o_blob_ = (blob*) o_blob;
		blob *s_blob_ = (blob*) s_blob;

		//echo();

	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo()  {
		LOG_INFO("create softmax op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blob->channel(), s_blob->width(), s_blob->height(),
				o_blob->channel(), o_blob->width(), o_blob->height());
	}

	inline void LOOP_INIT_DATA_()  {
		return;
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

private:

};
}


#endif
