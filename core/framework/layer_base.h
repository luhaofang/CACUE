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

	class layer_base
	{

	public:

		layer_base(int output_channel, int kernel_size, int stride, int pad, int input_dim, int channel){

			//_output_dim = _output_dim;
			_channel = channel;
			_input_dim = input_dim;
			_output_channel = output_channel;
			_kernel_size = kernel_size;
			_stride = stride;
			_pad = pad;
			_output_dim = 0;

			_args = new args(_output_channel,_kernel_size,_stride,_pad,_input_dim,_channel);

		};

		layer_base(){

			_args = NULL;

		};

		virtual ~layer_base(){
		
			delete _args;
		};


		//feature map output dim
		int _output_dim = 0;
		//input feature map channel
		int _channel = 0;
		//input dim
		int _input_dim = 0;
		//output feature map channel
		int _output_channel = 0;
		//kernel size
		int _kernel_size = 0;
		//padding size
		int _pad = 0;
		//stride size
		int _stride = 0;

		inline int op_count(){ return _ops.size(); }

		inline operator_base *&op(int i){ return _ops[i]; }

		inline void operate()
		{
			for(unsigned int i =0 ; i < _ops.size() ; ++i)
				_ops[i]->infer();
		}

	protected:

		vector<operator_base*> _ops;

		inline void add_op(operator_base *op)
		{
			_ops.push_back(op);
		}

		args_base *_args;

	private:



	};

};
