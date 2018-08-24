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

#ifndef FEATURE_COMBINE_OP_HPP_
#define FEATURE_COMBINE_OP_HPP_

namespace cacu {

class feature_combine_op: public operator_base {

public:

	feature_combine_op(blobs *&data, op_args *&args_) :
			operator_base(data, args_, CACU_FEATURE_COMBINE) {
		_INIT_OP();
	}

	~feature_combine_op() {

	}

	void initial()  {
		_units_count = _o_args->at(0);
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blob = create_em_oblob(s_blob->num() / _units_count,
					s_blob->channel() * _units_count, s_blob->width(),
					s_blob->height(), _phase);
#else
			o_blobs = create_oblobs();
			o_blobs->push_back(create_oblob(s_blobs->at(0)->num()/_units_count, s_blobs->at(0)->channel()*_units_count, s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else
			o_blobs->at(0)->resize(s_blobs->at(0)->num() / _units_count,
					s_blobs->at(0)->channel() * _units_count, s_blobs->at(0)->width(),
					s_blobs->at(0)->height());

	}

	void init_weights()  {
		return;
	}

	void check()  {
		if(_o_args == NULL)
			LOG_FATAL("feature combine op args cannot equal to NULL!");
		int mod = _o_args->at(0);
		mod = s_blobs->at(0)->num() % mod;
		CHECK_EQ_OP(mod, 0,
				"s_blob num must be integral multiple of units count vs %d!",
				mod);
		o_blobs->at(0)->_CHECK_SIZE_EQ(s_blobs->at(0));
	}

	void op()  {
#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		int output_num = s_blob->num() / _units_count;
		for (int i = 0; i < output_num; ++i) {
			for (int j = 0; j < _units_count; ++j) {
				cacu_copy_cpu(s_blob_->p_data(i * _units_count + j),
						s_blob_->length(),
						o_blob_->p_data(i) + j * s_blob_->length());
			}
		}
#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob_ = (blob*)s_blobs->at(0);

		int output_num = s_blobs->at(0)->num() / _units_count;
		for(int i = 0; i < output_num;++i)
		{
			for(int j = 0; j < _units_count; ++j)
			{
				cacu_copy(s_blob_->p_data(i*_units_count+j), s_blob_->length(), o_blob_->p_data(i)+j*s_blob_->length());
			}
		}
#endif
	}

	void grad()  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;

		int output_num = s_blob->num() / _units_count;
		for (int i = 0; i < output_num; ++i) {
			for (int j = 0; j < _units_count; ++j) {
				cacu_copy(o_blob_->p_diff_d(i) + j * s_blob_->length(),
						s_blob_->length(),
						s_blob_->p_diff_d(i * _units_count + j));
				s_blob_->_sync(i * _units_count + j);
			}
		}
#else
		blob *o_blob_ = (blob*)o_blobs->at(0);
		blob *s_blob_ = (blob*)s_blobs->at(0);

		int output_num = s_blobs->at(0)->num() / _units_count;
		for(int i = 0; i < output_num;++i)
		{
			for(int j = 0; j < _units_count; ++j)
			{
				cacu_copy(o_blob_->p_diff(i)+j*s_blob_->length(), s_blob_->length(), s_blob_->p_diff(i*_units_count+j));
			}
		}
#endif
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo() 
	{
		LOG_INFO("create feature combine op:");
		LOG_INFO(
				"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

private:

	//combine unit counts
	int _units_count = 0;
};
}

#endif
