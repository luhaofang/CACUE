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

#ifndef INJECTOR_OP_HPP_
#define INJECTOR_OP_HPP_

namespace cacu {

class injector_op: public operator_base {

public:

	injector_op(blobs *&data) :
			operator_base(data, CACU_INJECTOR) {
		_INIT_OP();
	}

	~injector_op() {

	}

	void initial()  {
		if (o_blobs == NULL) {
			o_blobs = s_blobs;
			_mean->resize(1, o_blobs->at(0)->channel(), 1, 1);
			_mask->resize(1, o_blobs->at(0)->channel(), o_blobs->at(0)->height(),
					o_blobs->at(0)->width());
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

		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_sumbysize(BYWIDTH, s_blob_->p_data_d(i), s_blob_->length(), 1,
					_mean->s_data(), 0, s_blob_->height() * s_blob_->width());
			cacu_scalex(_mean->s_data(), _mean->count(),
					(float_t) (1.0 / (s_blob_->height() * s_blob_->width())));
			cacu_ssxpy(_mean->s_data(), (float_t) (-1), _mean->count(),
					s_blob_->p_data_d(i), (float_t) 1, s_blob_->count(),
					o_blob_->p_data_d(i));
			cacu_relu(o_blob_ - p_data_d(i), o_blob_->length());
			mask_vector(o_blob_->p_data_d(i), o_blob_->length(),
					_mask->s_data());
			cacu_ssxpy(_mean->s_data(), (float_t) (1), _mean->count(),
					s_blob_->p_data_d(i), (float_t) 1, s_blob_->length(),
					o_blob_->p_data_d(i));
			cacu_ssx(_mask->s_data(), _mask->count(), s_blob_->p_data_d(i));
			o_blob_->_sync(i);
		}

#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob_ = (blob*)s_blobs->at(0);

		for(int i = 0; i < s_blob_->num(); ++i)
		{
			cacu_sumbysize(BYWIDTH,s_blob_->p_data(i),s_blob_->length(),1,_mean->s_data(),0,s_blob_->height()*s_blob_->width());
			cacu_scalex(_mean->s_data(),_mean->count(),((float_t)1.0/(s_blob_->height()*s_blob_->width())));
			cacu_ssxpy(_mean->s_data(),(float_t)(-1),_mean->count(),s_blob_->p_data(i),(float_t)1,s_blob_->length(),o_blob_->p_data(i));
			cacu_relu(o_blob_->p_data(i),o_blob_->length());
			mask_vector(o_blob_->p_data(i),o_blob_->length(),_mask->s_data());
			cacu_ssxpy(_mean->s_data(),(float_t)(1),_mean->count(),s_blob_->p_data(i),(float_t)1,s_blob_->length(),o_blob_->p_data(i));
			cacu_ssx(_mask->s_data(),_mask->count(), s_blob_->p_data(i));
		}
#endif
	}

	void grad()  {
		return;
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo() 
	{
		LOG_INFO("create injector op:");
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

	blob* _mean = NULL;
	blob* _mask = NULL;
};
}


#endif
