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

#pragma once

namespace cacu{

	class max_pooling_op : public operator_base
	{

	public:

		max_pooling_op(blob_base *&data, data_args *&args_) : operator_base(data, args_, CACU_MAX_POOLING){

			check();
			initial(data,args_);
			init_weights(data,args_);
			echo();

		}

		~max_pooling_op(){
			delete _index;
		}

		virtual const void initial(blob_base *&data, data_args *&args_) override{
			int input_w = data->width();
			int input_h = data->height();
			int channel = data->channel();
			int num = data->num();
			int output_w = (input_w - _args->kernel_size()) / _args->stride() + 1;
			int output_h = (input_h - _args->kernel_size()) / _args->stride() + 1;
			int pad = abs(input_w - (output_w - 1) * _args->stride() - _args->kernel_size());
			if (pad != 0)
				output_w += 1;
			pad = abs(input_h - (output_h - 1) * _args->stride() - _args->kernel_size());
			if (pad != 0)
				output_h += 1;

#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(num, channel, output_w, output_h, _phase);
			_index = cacu_allocator::create_em_bin_blob(num, channel, output_w, output_h, test);
#else
			o_blob = create_oblob(num, channel, output_w, output_h, _phase);
			_index = cacu_allocator::create_bin_blob(num, channel, output_w, output_h, test);
#endif

		}

		virtual const void init_weights(blob_base *&data, data_args *&args_) override{
			return;
		}

		virtual const void check() override{
			//kernel_size > 0
			CHECK_GT_OP(_args->kernel_size(), 0,"kernel_size must > 0 vs %d",_args->kernel_size());
			//CHECK_EQ_OP(_args->output_channel(),s_blob->channel(),"source data must equal to layer args output_channel!");
		}

		virtual const void op() override {
#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			em_bin_blob *index_ = (em_bin_blob*)_index;
			for(int i = 0 ; i < s_blob_->num(); ++i){
				cacu_max_pooling(s_blob_->p_data_d(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->height(), o_blob_->width(), o_blob_->height(), s_blob_->channel(), o_blob_->p_data_d(i), index_->p_data_d(i));
				o_blob_->_sync(i);
				index_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			bin_blob *index_ = (bin_blob*)_index;
			for(int i = 0 ; i < s_blob_->num(); ++i)
				cacu_max_pooling(s_blob_->p_data(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->height(), o_blob_->width(), o_blob_->height(), s_blob_->channel(), o_blob_->p_data(i), index_->p_data(i));
#endif

		}

		virtual const void grad() override{

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;
			em_bin_blob *index_ = (em_bin_blob*)_index;
			for(int i = 0 ; i < s_blob_->num(); ++i){
				cacu_max_pooling_grad(o_blob_->p_diff_d(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->height(), o_blob_->width(), o_blob_->height(), s_blob_->channel(), s_blob_->p_diff_d(i), index_->p_data_d(i));
				s_blob_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			bin_blob *index_ = (bin_blob*)_index;

			for(int i = 0 ; i < s_blob_->num(); ++i)
				cacu_max_pooling_grad(o_blob_->p_diff(i), _args->kernel_size(), _args->stride(), s_blob_->width(), s_blob_->height(), o_blob_->width(), o_blob_->height(), s_blob_->channel(), s_blob_->p_diff(i), index_->p_data(i));
#endif
		}

		virtual const void load(std::ifstream& is) override{
			return;
		}

		virtual const void save(std::ostream& os) override{
			return;
		}

		virtual const void echo() override{
			LOG_INFO("create max_pooling op:");
			LOG_INFO("channel: %d, input_dim: %d, output_channel: %d, output_dim: %d, kenrel_size: %d, stride: %d, pad: %d",s_blob->channel(),s_blob->height(),o_blob->channel(),o_blob->height(), _args->kernel_size(),_args->stride(),_args->pad());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();
			_index->_RESET_DATA();
		}

		inline virtual const void set_phase(phase_type phase_) override {
			_phase = phase_;
		}

	private:

		blob_base *_index;

	};
};
