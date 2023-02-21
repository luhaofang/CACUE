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

#ifndef CACU_OP_H_
#define CACU_OP_H_

#include "../ops/ops_definition.h"

#include "layer_base.h"

using namespace std;

namespace cacu {

class cacu_op {

public:

	/*
	 * cacu_op every operator contains the output space, including relu, dropout etc.
	 */
	cacu_op(op_name op_type_, phase_type phase);

	cacu_op(op_name op_type_, op_args *args_, phase_type phase);

	cacu_op(op_name op_type_, data_args *args_, phase_type phase);

	cacu_op(op_name op_type_, op_args *o_args_, data_args *args_, phase_type phase);

	cacu_op(chars_t&& op_name_, phase_type phase);

	cacu_op(chars_t&& op_name_, op_args *args_, phase_type phase);

	cacu_op(chars_t&& op_name_, data_args *args_, phase_type phase);

	cacu_op(chars_t&& op_name_, op_args *o_args_, data_args *args_, phase_type phase);

	~cacu_op();

	int caculate_data_space() {
		return 0;
	}

	template<typename OPTYPE>
	inline OPTYPE *get_op() const {
		if(_op_type == _op->_TYPE())
			return dynamic_cast<OPTYPE*>(_op);
		else{
			LOG_FATAL("Use a wrong type operator casting as %s!",
					MACRO_FACTORY_OP::get_cname(typeid(OPTYPE).name()).c_str());
		}
	}

	inline weight *get_param(int i) const {
		CHECK_LT_OP(i,_op->weights_size(),"parameter index is out of range %d vs %d!", i, _op->weights_size());
		return _op->get_weight(i);
	}

	inline vector<weight* > *get_weights() const {
		return _op->get_weights();
	}

	/**
	 * default data blob is [0]
	 */
	blobs* forward(blobs *datas_);

	void backward();

	inline void push2ins(cacu_op *op_)
	{
		_in_ops->push_back(op_);
	}

	inline void push2outs(cacu_op *op_)
	{
		_out_ops->push_back(op_);
	}

	inline blobs *out_datas() {
		return _out_datas;
	}

	inline blobs *in_datas() {
		return _in_datas;
	}

	//serialize calculation graphic node
	void serialize_node();

private:

	op_name _op_type;

	vector<cacu_op*> *_in_ops = NULL;

	//data maintained here
	blobs *_out_datas = NULL;

	//temp in_data blobs
	blobs *_in_datas = NULL;

	vector<cacu_op*> *_out_ops = NULL;

	operator_base *_op;

	phase_type _phase;

	void init_outdatas();

};

}


#endif
