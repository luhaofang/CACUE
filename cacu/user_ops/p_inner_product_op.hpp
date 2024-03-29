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

#ifndef P_INNER_PRODUCT_OP_HPP_
#define P_INNER_PRODUCT_OP_HPP_

namespace cacu {

class p_inner_product_op: public operator_base {

public:

	p_inner_product_op(blobs *&data, data_args *&args_) :
			operator_base(data, args_, CACU_P_INNERPRODUCT) {
		_INIT_OP();
	}

	~p_inner_product_op() {

	}

	void initial() override {
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_blobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), 1, 1,
					_phase));
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), _args->output_channel(), 1, 1, _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), _args->output_channel(), 1, 1);
		}
	}

	void init_weights() override {
		_w = create_param("w", s_blobs->at(0)->channel(), _args->output_channel() / s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
				s_blobs->at(0)->height(), _phase);

		_bias = create_param("bias", _args->output_channel(), 1, 1, 1, _phase);
		_bias->set_lr(2);
	}

	void check() override {
		CHECK_EQ_OP(_args->output_channel() % s_blobs->at(0)->channel(), 0, "Output data channel must integer times of input data channel: (%d)", _args->output_channel() % s_blobs->at(0)->channel());
	}

	void op(blobs *&s_blobs_,blobs *&o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		blob *temp_ = (blob*) _temp;

		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_copy(s_blob_->p_data(i), temp_->count(), temp_->s_data());
			cacu_ssx(_w->s_data(), _w->count(), temp_->s_data());
			cacu_sumbysize(BYWIDTH, temp_->s_data(), temp_->count(), 1,
					o_blob_->p_data(i), 0,
					s_blob_->height() * s_blob_->width());
			//bias added
			if (_is_use_bias) {
				cacu_ssxpy(_bias->s_data(), (float_t) (1), _bias->count(),
						o_blob_->p_data(i), (float_t) 1, o_blob_->length(),
						o_blob_->p_data(i));
			}
			o_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);

		for(int i = 0; i < s_blob_->num(); ++i) {

			for (int c = 0; c < s_blob_->channel(); ++c) 
				cacu_sgemm(TRANS, NOTRANS, _w->p_data(c), _w->channel(), _w->channel_length(), s_blob_->p_data(i) + c * s_blob_->channel_length(), 1, 1, o_blob_->p_data(i) + c * _w->channel(), 0);

			//bias added
			if (_is_use_bias)			
				cacu_saxpby(_bias->s_data(), (float_t)(1), o_blob_->p_data(i), (float_t)(1), _bias->count());
		}
#endif
	}

	void grad(blobs *&s_blobs_,blobs *&o_blobs_) override {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);
		em_blob *s_blob_ = (em_blob*) s_blobs->at(0);
		blob *temp_ = (blob*) _temp;

		for (int i = 0; i < s_blob_->num(); ++i) {
			//gradient propagation
			cacu_cxsize(_w->s_data(), _w->count(), o_blob_->p_diff(i),
					o_blob_->length(), s_blob_->p_diff(i));
			//weights gradient
			cacu_cxsize(s_blob_->p_data(i), s_blob_->length(),
					o_blob_->p_diff(i), o_blob_->length(), temp_->s_diff());
			cacu_saxpby(temp_->s_diff(), (float_t) 1, _w->s_diff(), (float_t) 1,
					_w->count());
			if (_is_use_bias)
				//bias gradient
				cacu_saxpby(o_blob_->p_diff(i), 1, _bias->s_diff(), 1,
						o_blob_->count());
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		blob *s_blob_ = s_blobs_->asblob(0);
			
		for (int i = 0; i < s_blob_->num(); ++i) {

			for (int c = 0; c < s_blob_->channel(); ++c) {
				//gradient propagation
				cacu_sgemm(NOTRANS, NOTRANS, _w->p_data(c), _w->channel_length(), _w->channel(), o_blob_->p_diff(i) + c * _args->output_channel() / s_blob_->channel(), 1, 1, s_blob_->p_diff(i) + c * s_blob_->channel_length(), 0);
				//weights gradient
				cacu_sgemm(NOTRANS, TRANS, s_blob_->p_data(i) + c * s_blob_->channel_length(), s_blob_->channel_length(), 1, o_blob_->p_diff(i) + c * _args->output_channel() / s_blob_->channel(), _args->output_channel() / s_blob_->channel(), 1, _w->p_diff(c), 1);
			}
		}
		if (_is_use_bias)
			//bias gradient
			cacu_sumbysize(BYHEIGHT, o_blob_->s_diff(), o_blob_->count(), 1, _bias->s_diff(), 1, _bias->count());
#endif
	}

	void load(std::ifstream& is) override {
		_w->load(is);
		if (_is_use_bias)
			_bias->load(is);
	}

	void save(std::ostream& os) override {
		_w->serializa(os);
		if (_is_use_bias)
			_bias->serializa(os);
	}

	void echo() override
	{
		LOG_INFO("create p_inner_product op:");
		LOG_INFO(
				"channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",
				s_blobs->at(0)->channel(), s_blobs->at(0)->height(), o_blobs->at(0)->channel(),
				o_blobs->at(0)->height());
	}

	inline void set_weight_init_type(param_init_type _type,
			float_t value = 0.0) {
		set_param_init_type(_type, _w, value);
	}

	inline void set_bias_init_type(param_init_type _type, float_t value = 0.0) {
		set_param_init_type(_type, _bias, value);
	}

	void is_use_bias(bool switcher_) {
		_is_use_bias = switcher_;
	}

private:

	//p_innerproduct_op use bias switcher
	bool _is_use_bias = true;

	weight *_w = NULL;

	weight *_bias = NULL;

};
}


#endif
