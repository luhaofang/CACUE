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

#ifndef TANH_OP_HPP_
#define TANH_OP_HPP_

namespace cacu {

class tanh_op: public operator_base {

public:

	tanh_op(blobs *&data) :
			operator_base(data, CACU_TANH) {
		_INIT_OP();
	}

	~tanh_op() {

	}

	void initial()  {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->height(), s_blobs->at(0)->width(), _phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(),
					s_blobs->at(0)->width(), s_blobs->at(0)->height());
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
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_tanh_cpu(s_blob_->s_data(), o_blob_->count(), o_blob_->s_data());

#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob_ = (blob*)s_blobs->at(0);

		cacu_tanh(s_blob_->s_data(), o_blob_->count(), o_blob_->s_data());
#endif
	}

	void grad()  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

		cacu_tanh_grad_cpu(o_blob_->s_data(), o_blob_->s_diff(),
				s_blob_->count(), s_blob_->s_diff());

#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob_ = (blob*)s_blobs->at(0);

		cacu_tanh_grad(o_blob_->s_data(), o_blob_->s_diff(), s_blob_->count(), s_blob_->s_diff());

#endif
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo()  {
		LOG_INFO("create tanh op:");
		LOG_INFO(
				"channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",
				s_blobs->at(0)->channel(), s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
				o_blobs->at(0)->height());
	}

	inline void LOOP_INIT_DATA_() 
	{
		return;
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

private:

};
}


#endif
