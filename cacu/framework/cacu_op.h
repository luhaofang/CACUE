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

namespace cacu {

class cacu_op {

public:

	cacu_op(op_name op_type_);

	cacu_op(op_name op_type_, op_args *args_);

	cacu_op(op_name op_type_, blob_base *s_data_, op_args *args_);

	cacu_op(op_name op_type_, blob_base *s_data_, data_args *args_);

	~cacu_op();

	int caculate_data_space() {
		return 0;
	}

	template<typename OPTYPE>
	inline OPTYPE *&get_op() const {
		if(_op_type == _op->_TYPE())
			return (OPTYPE*&)_op;
		else{
			LOG_FATAL("Shot! You are using a wrong type operator casting!");
		}
	}

	cacu_op* operator <<(cacu_op *&op_);

	cacu_op* operator >>(cacu_op *&op_);

	cacu_op* operator <<(blob_base *&data_);

	template<typename BTYPE>
	inline BTYPE *&get_sblob() {
		return _op->in_data<BTYPE>();
	}

	template<typename BTYPE>
	inline BTYPE *&get_oblob() {
		return _op->out_data<BTYPE>();
	}

	inline blobs *&get_sblobs() {
		return _op->in_datas();
	}

	inline blobs *&get_oblobs() {
		return _op->out_datas();
	}

	inline weight *get_param(int i)
	{
		CHECK_LT_OP(i,_op->weights_size(),"parameter index is out of range %d vs %d!", i, _op->weights_size());
		return _op->get_weight(i);
	}

	inline void run()
	{
		_op->infer();
	}

	/**
	 * assign operator
	 * this function is called after the calculation steps are assigned.
	 * every input of the
	 * WARNING: focus on the multi output operator such as split_op.
	 */
	void _CREATE_OP(){
		for(int i = 0 ; i < _INs->size(); ++i) {
			_op->in_datas()->push_back(_INs->at(i)->get_oblob<blob_base>());
			_INs->at(i)->get_oblob<blob_base>()->_REC();
		}
		//initial phase
		_op->init_sblob();
		//initial operator
		_op->alloc_create_op();
	}

private:

	op_name _op_type;

	vector<cacu_op*> *_INs = NULL;

	operator_base *_op;

};

}


#endif
