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

#include "cacu_op.h"

namespace cacu {

cacu_op::cacu_op(op_name op_type_, phase_type phase_)
{
	_op_type = op_type_;
	blobs *s_datas_ = new blobs();
	s_datas_->push_back(new blob(1,1,1,1,0,phase_,false));
	_op = operator_factory::create_op(op_type_, s_datas_, NULL, NULL);
	_in_ops = new vector<cacu_op*>();
	_out_ops = new vector<cacu_op*>();
	_out_datas = _op->out_datas()->copy_create();
	_phase = phase_;
}

cacu_op::cacu_op(op_name op_type_, op_args *args_, phase_type phase_)
{
	_op_type = op_type_;
	blobs *s_datas_ = new blobs();
	s_datas_->push_back(new blob(1,1,1,1,0,phase_,false));
	_op = operator_factory::create_op(op_type_, s_datas_, NULL, args_);
	_in_ops = new vector<cacu_op*>();
	_out_ops = new vector<cacu_op*>();
	_out_datas = _op->out_datas()->copy_create();
	_phase = phase_;
}

cacu_op::cacu_op(op_name op_type_, data_args *args_, phase_type phase_)
{
	_op_type = op_type_;
	blobs *s_datas_ = new blobs();
	s_datas_->push_back(new blob(1,1,1,1,0,phase_,false));
	_op = operator_factory::create_op(op_type_, s_datas_, args_, NULL);
	_in_ops = new vector<cacu_op*>();
	_out_ops = new vector<cacu_op*>();
	_out_datas = _op->out_datas()->copy_create();
	_phase = phase_;
}

/*
 * cacu_op don't maintain the input operator list.
 * don't need to release input cacu_op memory
 */
cacu_op::~cacu_op()
{
	delete _in_ops;
	_in_ops = NULL;
	delete _out_ops;
	_out_ops = NULL;
	delete _op;
	_op = NULL;
	delete _out_datas;
	_out_datas = NULL;
}


/**
 * default data blob is [0]
 */
blobs *cacu_op::forward(blobs *&datas_)
{
	/*
	 * fix op output blob body
	 */
	blob_base *sblob_op = _op->in_datas()->at(0);
	blob_base *sblob_ = datas_->at(0);
	if (!sblob_->body()->check_body(sblob_op->body())) {
		sblob_op->body()->copy_from(sblob_->body());
		_op->initial();
	}
	//reset the data's values
	_op->LOOP_INIT_DATA_();
	//if op alloc the output space, resize out data
	if(_op->is_alloc_output()){
		_out_datas->resize(_op->out_data<blob>()->body());
		_out_datas->_RESET_DATA();
	}
	//if op don't need to alloc output space, just confirm the output size;
	else{
		_out_datas = datas_;
	}
	//need reverse propagate, here to storage the input data
	_op->op(datas_, _out_datas);
	_in_datas = datas_;
	return _out_datas;
}

void cacu_op::backward()
{
	if(_phase == train)
		_op->grad(_in_datas, _out_datas);
}


}
