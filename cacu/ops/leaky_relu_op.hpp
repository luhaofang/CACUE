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

#ifndef LEAKY_RELU_OP_HPP_
#define LEAKY_RELU_OP_HPP_

namespace cacu {

class leaky_relu_op: public operator_base {

public:

	leaky_relu_op(blobs *&data, op_args *&args_) :
			operator_base(data, args_, CACU_LEAKY_RELU) {
		_INIT_OP();
	}

	~leaky_relu_op() {

	}

	void initial()  {
		_negative_slope = _o_args->at(0);
		if (o_blobs == NULL)
			o_blobs = s_blobs;
		else
			o_blobs->_NEED_MOTIFY();
	}

	void init_weights()  {
		return;
	}

	void check()  {
		if(_o_args == NULL)
			LOG_FATAL("leaky relu op args cannot equal to NULL!");
		//negative_slope > 0
		CHECK_GT_OP(_o_args->at(0), 0, "negative slope must > 0 vs %d",
				_o_args->at(0));
	}

	void op(blobs *s_blobs_,blobs *o_blobs_)  {

		o_blobs_ = s_blobs_;
		blob *s_blob_ = (blob*) s_blobs_->at(0);

		cacu_leaky_relu(s_blob_->s_data(), _negative_slope, s_blob_->count());
	}

	void grad(blobs *s_blobs_,blobs *o_blobs_)  {
		blob *o_blob_ = (blob*) o_blobs_->at(0);
		blob *s_blob_ = (blob*) s_blobs_->at(0);
		cacu_leaky_relu_grad(s_blob_->s_data(), o_blob_->s_diff(),
				_negative_slope, s_blob_->count());
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo()  {
		LOG_INFO("create leaky_relu op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

	float_t _negative_slope = 0.01;

private:

};
}


#endif
