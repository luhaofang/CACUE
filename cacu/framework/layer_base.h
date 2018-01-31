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
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS size_tERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "args_base.h"
#include "data_args.h"
#include "../ops/operator_base.h"

#include "../factory/operator_factory.h"

namespace cacu {

class layer_base {

public:

	layer_base(size_t output_channel, size_t kernel_size, size_t stride, size_t pad,
			size_t input_dim, size_t channel) {

		//_output_dim = _output_dim;
		_channel = channel;
		_input_dim = input_dim;
		_output_channel = output_channel;
		_kernel_size = kernel_size;
		_stride = stride;
		_pad = pad;
		_output_dim = 0;

		_args = new data_args(_output_channel, _kernel_size, _stride, _pad,
				_input_dim, _channel);

		_ops = new vector<operator_base*>();

	}

	layer_base() {
		_args = NULL;
		_ops = new vector<operator_base*>();
	}

	virtual ~layer_base() {

		delete _args;
		for(int i =0 ; i < _ops->size(); ++i)
		{
			delete _ops->at(i);
			_ops->at(i)= NULL;
		}
		delete _ops;
	}

	//feature map output dim
	size_t _output_dim = 0;
	//input feature map channel
	size_t _channel = 0;
	//input dim
	size_t _input_dim = 0;
	//output feature map channel
	size_t _output_channel = 0;
	//kernel size
	size_t _kernel_size = 0;
	//padding size
	size_t _pad = 0;
	//stride size
	size_t _stride = 0;

	inline size_t op_count() const {
		return _ops->size();
	}

	inline operator_base* op(size_t i) {
		return _ops->at(i);
	}

	inline void operate() {
		for (unsigned int i = 0; i < _ops->size(); ++i) {
			_ops->at(i)->infer();
			//LOG_DEBUG("op: %d", i);
		}
	}

	inline void grad() {
		for (int i = _ops->size() - 1; i >= 0; --i) {
			_ops->at(i)->grad();
			//LOG_DEBUG("op: %d", i);
		}
	}

protected:

	vector<operator_base*> *_ops;

	inline void add_op(operator_base *op) {
		_ops->push_back(op);
	}

	data_args* _args;

private:

};

}
