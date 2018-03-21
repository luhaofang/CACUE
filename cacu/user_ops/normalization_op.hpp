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

#ifndef NORMALIZATION_OP_HPP_
#define NORMALIZATION_OP_HPP_


namespace cacu {

class normalization_op: public operator_base {

public:

	normalization_op(blob_base *&data, data_args *&args_) :
			operator_base(data, args_, CACU_NORM) {

		check();
		initial();
		init_weights();
		echo();

	}

	~normalization_op() {

	}

	virtual const void initial() override {
		if (o_blob == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(s_blob->num(), s_blob->channel(),
					s_blob->height(), s_blob->width(), _phase);
			_temp = create_em_opblob(1, s_blob->channel(), s_blob->height(),
					s_blob->width(), 1.0, _phase);
#else
			o_blob = create_oblob(s_blob->num(), s_blob->channel(), s_blob->height(), s_blob->width(), _phase);
			_temp = create_opblob(1, s_blob->channel(), s_blob->height(),s_blob->width(), 1.0, _phase);
#endif
		} else {
			o_blob->resize(s_blob->num(), s_blob->channel(), s_blob->height(),
					s_blob->width());
			_temp->resize(1, s_blob->channel(), s_blob->height(),
					s_blob->width(), 1.0);
		}
	}

	virtual const void init_weights() override {
		return;
	}

	virtual const void check() override {

	}

	virtual const void op() override {
#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_blob *temp_ = (em_blob*) _temp;

		for (int i = 0; i < s_blob_->num(); ++i) {
			//cacu_max_pooling(s_blob_->p_data_d(i), _args->kernel_size(), _args->stride(), s_blob_->width(), o_blob_->width(), s_blob_->channel(), o_blob_->p_data_d(i), index_->p_data_d(i));
			o_blob_->_sync(i);
			temp_->_sync(i);
		}
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;
		blob *temp_ = (blob*)_temp;

		switch(_NORMALIZER)
		{
			case L1:
			for(int i = 0; i < s_blob_->num(); ++i) {
				//cacu_max_pooling(s_blob_->p_data(i), _args->kernel_size(), _args->stride(), s_blob_->width(), o_blob_->width(), s_blob_->channel(), o_blob_->p_data(i), index_->p_data(i));
			}
			break;
			case L2:
			for(int i = 0; i < s_blob_->num(); ++i) {
				//cacu_max_pooling(s_blob_->p_data(i), _args->kernel_size(), _args->stride(), s_blob_->width(), o_blob_->width(), s_blob_->channel(), o_blob_->p_data(i), index_->p_data(i));
			}
			break;
			default:
			break;
		}

#endif

	}

	virtual const void grad() override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_bin_blob *temp_ = (em_bin_blob*) _temp;
		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_max_pooling_grad(o_blob_->p_diff_d(i), _args->kernel_size(),
					_args->stride(), s_blob_->width(), o_blob_->width(),
					s_blob_->channel(), s_blob_->p_diff_d(i),
					index_->p_data_d(i));
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;
		bin_blob *temp_ = (bin_blob*)_temp;

		//for(int i = 0 ; i < s_blob_->num(); ++i)
		//cacu_max_pooling_grad(o_blob_->p_diff(i), _args->kernel_size(), _args->stride(), s_blob_->width(), o_blob_->width(), s_blob_->channel(), s_blob_->p_diff(i), index_->p_data(i));

#endif
	}

	virtual const void load(std::ifstream& is) override {
		return;
	}

	virtual const void save(std::ostream& os) override {
		return;
	}

	virtual const void echo() override {
		LOG_INFO("create max_pooling op:");
		LOG_INFO(
				"channel: %d, input_dim: %d, output_channel: %d, output_dim: %d, kenrel_size: %d, stride: %d, pad: %d",
				s_blob->channel(), s_blob->height(), o_blob->channel(),
				o_blob->height(), _args->kernel_size(), _args->stride(),
				_args->pad());
	}

	inline virtual const void LOOP_INIT_DATA_() override
	{
		o_blob->_RESET_DATA();
	}

	inline virtual const void set_phase(phase_type phase_) override {
		_phase = phase_;
	}

private:

	blob *_temp = NULL;

	normalize_type _NORMALIZER = norm_l2;

};
}


#endif