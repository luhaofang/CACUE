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


#include "../ops/operator_factory.h"

namespace mycnn{

	class rnn_cell_unit : public layer_base{


	public:

		rnn_cell_unit(int time_seq, blob_base *in_data, int output_channel, int kernel_size, int stride, int pad)
			:layer_base(output_channel, kernel_size, stride, pad, in_data->width(), in_data->channel()){

			in_blobs = cacu_allocator::create_blobs();
			for(int i = 0;i < time_seq; ++i)
				in_blobs->push_back(cacu_allocator::create_blob(in_data));

			out_blobs = cacu_allocator::create_blobs();

			//hidden layer weights
			h_w = new weight("w", output_channel,output_channel,1,1,in_data->phrase());
			h_bias = new weight("bias", output_channel,1,1,1,in_data->phrase());

			mix_op = operator_factory::create_op(CACU_INNERPRODUCT, out_blobs->front(), (args*&)_args);

			_time_seq = time_seq;

		};


		~rnn_cell_unit(){

			delete in_blobs;
			delete out_blobs;

			delete h_w;
			delete h_bias;

			delete mix_op;

		};


		layer* op(op_name op_) {

			blobs *blobs_ = cacu_allocator::create_blobs();
			if (out_blob != NULL)
				blobs_->push_back(out_blob);
			add_op(operator_factory::create_op(op_, blobs_, (args*&)_args));
			out_blob = _ops.back()->out_data<blob_base>();

			out_blobs->_DELETE_BLOBS();
			for(int i = 0;i < time_seq; ++i)
				out_blobs->push_back(cacu_allocator::create_blob(out_blob));

			return this;
		}

		layer* op(op_name op_, blob_base *blob_) {
			if(blob_ == NULL)
				LOG_FATAL("input data is NULL!");
			refresh_layer_param(blob_);
			blobs *blobs_ = cacu_allocator::create_blobs();
			if (out_blob != NULL)
				blobs_->push_back(out_blob);
			blobs_->push_back(blob_);
			add_op(operator_factory::create_op(op_, blobs_, (args*&)_args));
			out_blob = _ops.back()->out_data<blob_base>();

			out_blobs->_DELETE_BLOBS();
			for(int i = 0;i < time_seq; ++i)
				out_blobs->push_back(cacu_allocator::create_blob(out_blob));

			return this;
		}

		layer* op(op_name op_, blob_base *blob_ ,args_base *args_) {
			if(blob_ == NULL)
				LOG_FATAL("input data is NULL!");
			_args = args_;
			blobs *blobs_ = cacu_allocator::create_blobs();
			if (out_blob != NULL)
				blobs_->push_back(out_blob);
			blobs_->push_back(blob_);
			add_op(operator_factory::create_op(op_, blobs_, (args*&)_args));
			out_blob = _ops.back()->out_data<blob_base>();

			out_blobs->_DELETE_BLOBS();
			for(int i = 0;i < time_seq; ++i)
				out_blobs->push_back(cacu_allocator::create_blob(out_blob));

			return this;
		}

		layer* op(op_name op_, blobs *blobs_) {
			if(blobs_ == NULL)
				LOG_FATAL("input data is NULL!");
			refresh_layer_param(blobs_->at(0));

			if (out_blob != NULL)
				blobs_->push_back(out_blob);
			add_op(operator_factory::create_op(op_, blobs_, (args*&)_args));
			out_blob = _ops.back()->out_data<blob_base>();

			out_blobs->_DELETE_BLOBS();
			for(int i = 0;i < time_seq; ++i)
				out_blobs->push_back(cacu_allocator::create_blob(out_blob));

			return this;
		}

		layer* op(op_name op_, blobs *blobs_, args_base *args_) {
			if(blobs_ == NULL)
				LOG_FATAL("input data is NULL!");
			_args = args_;
			if (out_blob != NULL)
				blobs_->push_back(out_blob);
			add_op(operator_factory::create_op(op_, blobs_, (args*&)_args));
			out_blob = _ops.back()->out_data<blob_base>();

			out_blobs->_DELETE_BLOBS();
			for(int i = 0;i < time_seq; ++i)
				out_blobs->push_back(cacu_allocator::create_blob(out_blob));

			return this;
		}

		inline void operate() override
		{

			for(unsigned int t = 0; t < _time_seq; ++t)
			{
				//forward input blob
				copy2indata();
				for(unsigned int i = 0 ; i < _ops.size() ; ++i){
					_ops[i]->infer();
				}
				//forward output blob
				if(t == 0){
					copy2outdata();
				}
				else{
					//get last time output data
					mix_op->set_blob(get_lasttime_oblob());
					mix_op->infer();
					cacu_saxpy(((blob *)mix_op->out_data())->s_data(),1,op_oblob()->s_data(),op_oblob()->count());
					copy2outdata();
				}
			}
		}

		inline void grad() override
		{
			for(unsigned int t = 0; t < _time_seq; ++t){
				copy2outdiff();

				for(int i = _ops.size() - 1 ; i >= 0; --i)
				{
					_ops[i]->grad();
				}


			}
		}

		template<class OPTYPE>
		inline OPTYPE *& get_op(int i)
		{
			return (OPTYPE*&)_ops[i];
		}

		inline blob *& op_oblob()
		{
			return _ops.back()->out_data<blob>();
		}

		inline blob *& op_iblob()
		{
			return _ops.front()->in_data<blob>();
		}

		inline blobs *&get_oblobs()
		{
			return out_blobs;
		}

		inline blob_base* get_input(int index){

			CHECK_LT(index, "index must <= %d vs %d!",_index , index);

			return in_blobs->at(index);
		}

		inline blob_base* get_output(int index){

			CHECK_LT(index, "index must <= %d vs %d!",_index , index);

			return out_blobs->at(index);
		}





	protected:

		inline void copy2indata()
		{
			blob *inblob = (blob*)in_blobs->front();
			cacu_copy(inblob->s_data(), inblob->count(), op_iblob()->s_data());
			_it = in_blobs->begin();
			in_blobs->erase(_it);
			in_blobs->push_back(inblob);
		}


		inline void copy2outdata()
		{
			blob *outblob = (blob*)out_blobs->front();
			cacu_copy(op_oblob()->s_data(), outblob->count(), outblob->s_data());
			_it = out_blobs->begin();
			out_blobs->erase(_it);
			out_blobs->push_back(outblob);
		}

		inline void copy2indiff()
		{
			blob *inblob = (blob*)in_blobs->front();
			cacu_copy(inblob->s_diff(), inblob->count(), op_iblob()->s_diff());
			_it = in_blobs->begin();
			in_blobs->erase(_it);
			in_blobs->push_back(inblob);
		}


		inline void copy2outdiff()
		{
			blob *outblob = (blob*)out_blobs->front();
			cacu_copy(op_oblob()->s_diff(), outblob->count(), outblob->s_diff());
			_it = out_blobs->begin();
			out_blobs->erase(_it);
			out_blobs->push_back(outblob);
		}

		inline blob_base* get_lasttime_oblob() { return out_blobs->back(); }

		inline blob_base* get_lasttime_iblob() { return in_blobs->back(); }


	private:

		blobs* out_blobs = NULL;

		blobs* in_blobs = NULL;

		weight* h_w;

		weight* h_bias;

		inner_product_op *mix_op;

		unsigned int _index = 0;

		unsigned int _time_seq = 0;

		//use for selecting blob_base
		blobs::iterator _it;

		//args refresh by input blob
		inline void refresh_layer_param(blob_base *blob_)
		{
			_args->at(4) = blob_->height();
			_args->at(5) = blob_->channel();
		}

	};

};
