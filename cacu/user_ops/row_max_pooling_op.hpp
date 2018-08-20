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

#ifndef ROW_MAX_POOLING_OP_HPP_
#define ROW_MAX_POOLING_OP_HPP_

namespace cacu {

class row_max_pooling_op: public operator_base {

public:

	row_max_pooling_op(blobs *&data, data_args *&args_) :
			operator_base(data, args_, CACU_ROW_MAX_POOLING) {
		_INIT_OP();
	}

	~row_max_pooling_op() {

	}

	void initial()  {

		int input_dim = s_blob->width();
		int channel = s_blob->channel();
		int num = s_blob->num();
		int output_length = s_blob->length() / _args->at(0);
		if (o_blob == NULL) {
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(num, output_channel, 1, 1, _phase);
			_index = create_em_bin_opblob(num, output_length, 1, 1, test);
			_x = create_opblob(1, channel, input_dim, input_dim, test);
#else
			o_blob = create_oblob(num, output_length, 1, 1, _phase);
			_index = create_bin_opblob(num, output_length, 1, 1, test);
			_x = create_opblob(1, channel, input_dim, input_dim, test);
#endif
		} else {
			o_blob->resize(num, output_length, 1, 1);
			_index->resize(num, output_length, 1, 1);
			_x->resize(1, channel, input_dim, input_dim);
		}

	}

	void init_weights()  {
		return;
	}

	void check()  {
		//kernel_size > 0
		CHECK_GT_OP(_args->at(0), 0, "output_channel must > 0 vs %d",
				_args->at(0));
	}

	void op()  {
#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_bin_blob *index_ = (em_bin_blob*) _index;
		em_blob *x_ = (em_blob*) _x;
		if (_phase == train) {
			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_copy(s_blob_->p_data_d(i), s_blob_->length(),
						x_->s_data());
				cacu_row_max_pooling(x_->s_data(), x_->count(),
						o_blob_->length(), o_blob_->p_data_d(i));
				cacu_row_max_pooling_index(s_blob_->p_data_d(i),
						s_blob_->length(), o_blob_->length(),
						o_blob_->p_data_d(i), index_->p_data_d(i));
				index_->_sync(i);
			}
		} else
			for (int i = 0; i < s_blob_->num(); ++i) {
				cacu_row_max_pooling(s_blob_->p_data_d(i), s_blob_->length(),
						o_blob_->length(), o_blob_->p_data_d(i));
				o_blob_->_sync(i);
			}
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;
		blob *x_ = (blob*)_x;
		bin_blob *index_ = (bin_blob*)_index;
		if(_phase == train) {
			for(int i = 0; i < s_blob_->num(); ++i) {
				cacu_copy(s_blob_->p_data(i),s_blob_->length(),x_->s_data());
				cacu_row_max_pooling(x_->s_data(),x_->count(),o_blob_->length(),o_blob_->p_data(i));
				cacu_row_max_pooling_index(s_blob_->p_data(i),s_blob_->length(),o_blob_->length(),o_blob_->p_data(i),index_->p_data(i));
			}
		}
		else
		for(int i = 0; i < s_blob_->num(); ++i)
		cacu_row_max_pooling(s_blob_->p_data(i),s_blob_->length(),o_blob_->length(),o_blob_->p_data(i));
#endif
	}

	void grad()  {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blob;
		em_blob *s_blob_ = (em_blob*) s_blob;
		em_bin_blob *index_ = (em_bin_blob*) _index;
		for (int i = 0; i < s_blob_->num(); ++i) {
			cacu_row_max_pooling_grad(o_blob_->p_diff_d(i), o_blob_->length(),
					s_blob_->p_diff_d(i), index_->p_data_d(i));
			s_blob_->_sync(i);
		}
#else
		blob *o_blob_ = (blob*)o_blob;
		blob *s_blob_ = (blob*)s_blob;
		bin_blob *index_ = (bin_blob*)_index;

		for(int i = 0; i < s_blob_->num(); ++i)
		cacu_row_max_pooling_grad(o_blob_->p_diff(i), o_blob_->length(), s_blob_->p_diff(i), index_->p_data(i));
#endif
	}

	void load(std::ifstream& is)  {
		return;
	}

	void save(std::ostream& os)  {
		return;
	}

	void echo()  {
		LOG_INFO("create row_max_pooling op:");
		LOG_INFO("channel: %d, input_dim: %d, output_length: %d",
				s_blob->channel(), s_blob->height(), o_blob->length());
	}

	inline void LOOP_INIT_DATA_() 
	{
		o_blob->_RESET_DATA();
		_index->_RESET_DATA();
	}

	inline void set_phase(phase_type phase_)  {
		_phase = phase_;
	}

private:

	bin_blob *_index = NULL;

	blob *_x = NULL;

};
}


#endif
