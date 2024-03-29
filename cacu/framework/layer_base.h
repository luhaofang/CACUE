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

#ifndef LAYER_BASE_H_
#define LAYER_BASE_H_

#include "args_base.h"
#include "data_args.h"
#include "op_args.h"

#include "../factory/operator_factory.h"

namespace cacu {

class layer_base {

public:

	layer_base(data_args *&args_) {

		_args = args_;
		_ops = new vector<operator_base*>();
		out_blob = NULL;

	}

	layer_base() {
		_args = NULL;
		out_blob = NULL;
		_ops = new vector<operator_base*>();
	}

	virtual ~layer_base() {
		delete _ops;
		_ops = NULL;

		_args = NULL;
		out_blob = NULL;
	}


	inline dsize_t op_count() const {
		return _ops->size();
	}

	inline operator_base* get_op_base(dsize_t i) {
		return _ops->at(i);
	}

	inline void operate() {
		for (unsigned i = 0; i < _ops->size(); ++i) {
			//LOG_DEBUG("op %d: %d",_ops->at(i)->get_optype(), i);
			_ops->at(i)->infer();
			//LOG_DEBUG("%d", i);
			//cacu_print(_ops->at(i)->out_data<blob>()->s_data(),100);
		}
	}

	inline void grad() {
		for (int i = (_ops->size()) - 1; i >= 0; --i) {
			_ops->at(i)->derivative();
			//cacu_print(_ops->at(i)->in_data<blob>()->s_diff(),100);
			//LOG_DEBUG("op: %d", i);
		}
	}

	inline void load_weights(std::ifstream &is)
	{
		for(int i = 0 ; i < _ops->size(); ++i)
		{
			_ops->at(i)->load(is);
		}
	}

	inline void load_weights_reverse(std::ifstream &is)
	{
		for(int i = _ops->size() - 1 ; i >= 0; --i)
		{
			_ops->at(i)->load(is);
		}
	}

	inline void save_weights(std::ofstream &os)
	{
		for(int i = 0 ; i < _ops->size(); ++i)
		{
			_ops->at(i)->save(os);
		}
	}

protected:

	vector<operator_base*> *_ops;

	inline void add_op(operator_base *op) {
		_ops->push_back(op);
	}

	data_args* _args;

	blob_base* out_blob = NULL;

};

}


#endif
