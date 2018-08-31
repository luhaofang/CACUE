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

#ifndef ROI_POOLING_OP_HPP_
#define ROI_POOLING_OP_HPP_

namespace cacu {

class roi_pooling_op: public operator_base {

public:

	roi_pooling_op(blobs *&data, op_args *&o_args_) :
			operator_base(data, o_args_, CACU_ROI_POOLING) {
		_INIT_OP();
	}

	~roi_pooling_op() {

	}

	void initial()  {

		int input_dim = s_blobs->at(0)->width();
		int channel = s_blobs->at(0)->channel();
		int num = s_blobs->at(0)->num();
		int output_dim = (input_dim - _args->kernel_size()) / _args->stride()
				+ 1;
		int pad = abs(
				input_dim - (output_dim - 1) * _args->stride()
						- _args->kernel_size());
		if (pad != 0)
			output_dim += 1;
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o+blobs = create_em_blobs();
			o_blobs->push_back(create_em_oblob(num, channel, output_dim, output_dim,
					_phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(num, channel, output_dim, output_dim, _phase));
#endif
		} else {
			o_blobs->at(0)->resize(num, channel, output_dim, output_dim);
		}
	}

	void init_weights()  {
		return;
	}

	void check()  {
		if(_o_args == NULL)
			LOG_FATAL("roipooling op args cannot equal to NULL!");
		int ph = _o_args->at(0);
		int pw = _o_args->at(1);
		int spatial_scale = _o_args->at(2);
		//kernel_size > 0
		CHECK_GT_OP(ph, 0, "pool_h must > 0 vs %d",
				ph);
		CHECK_GT_OP(pw, 0, "pool_w must > 0 vs %d",
				pw);
		CHECK_GT_OP(spatial_scale, 0, "spatial_scale must > 0 vs %d",
				spatial_scale);
	}

	void op(blobs *s_blobs_,blobs *o_blobs_)  {
#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);

#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);

#endif

	}

	void grad(blobs *s_blobs_,blobs *o_blobs_)  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

#else
		blob *o_blob_ = (blob*)o_blobs_->at(0);
		blob *s_blob_ = (blob*)s_blobs_->at(0);

#endif
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo()  {
		LOG_INFO("create max_pooling op:");
		LOG_INFO(
				"channel: %d, input_dim: %d, output_channel: %d, output_dim: %d, kenrel_size: %d, stride: %d, pad: %d",
				s_blobs->at(0)->channel(), s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
				o_blobs->at(0)->height(), _args->kernel_size(), _args->stride(),
				_args->pad());
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

private:




};
}


#endif
