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

namespace mycnn{

	class max_pooling_op : public operator_base
	{

	public:

		max_pooling_op(blob *&data, args *&args_) : operator_base((blob_base *&)data, args_){
			check();

			int input_dim = data->width();
			int channel = data->channel();
			int num = data->num();
			int output_dim = (input_dim - args_->at(1)) / args_->at(2) + 1;
			if ((input_dim - args_->at(1)) % args_->at(2) != 0)
				output_dim += 1;			
			o_blob = cacu_allocator::create_blob(num, channel, output_dim, output_dim);

			_index = cacu_allocator::create_bin_blob(num, channel, output_dim, output_dim);

		};

		~max_pooling_op(){
			delete (blob *)o_blob;
			delete _index;
		};

		virtual const void check() override{
			return;
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;
			cacu_max_pooling(s_blob_->s_data(), _args->at(1), _args->at(2), s_blob_->width(), o_blob_->width(), s_blob_->channel(), o_blob_->s_data(), _index->s_data());
			echo();
			return;
		}

		virtual const void grad(const solver_base *&solver_base) override{

		}

		virtual const void load(std::ifstream& is){

		}

		virtual const void save(std::ostream& os){

		}

		virtual const void echo()
		{
			//LOG_INFO("%f", ((blob*)o_blob)->s_data()[0]);
		}

	private:

		bin_blob *_index;

	};
};
