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

cacu_op::cacu_op(op_name op_type_)
{
	_op_type = op_type_;
	blobs *_s_datas = new blobs();
	_op = operator_factory::create_op(op_type_, _s_datas, NULL, NULL);
	_INs = new vector<cacu_op*>();
}

cacu_op::cacu_op(op_name op_type_, op_args *args_)
{
	_op_type = op_type_;
	blobs *_s_datas = new blobs();
	_op = operator_factory::create_op(op_type_, _s_datas, NULL, args_);
	_INs = new vector<cacu_op*>();
}

cacu_op::cacu_op(op_name op_type_, blob_base *s_data_, op_args *args_)
{
	_op_type = op_type_;
	blobs *_s_datas = new blobs();
	_s_datas->push_back(s_data_);
	_op = operator_factory::create_op(op_type_, _s_datas, NULL, args_);
	_INs = new vector<cacu_op*>();
}

cacu_op::cacu_op(op_name op_type_, blob_base *s_data_, data_args *args_)
{
	_op_type = op_type_;
	blobs *_s_datas = new blobs();
	_s_datas->push_back(s_data_);
	_op = operator_factory::create_op(op_type_, _s_datas, args_, NULL);
	_INs = new vector<cacu_op*>();
}

/*
 * cacu_op don't maintain the input operator list.
 * don't need to release input cacu_op memory
 */
cacu_op::~cacu_op()
{
	delete _INs;
	_INs = NULL;
	delete _op;
	_op = NULL;
}


cacu_op* cacu_op::operator << (cacu_op *&cacu_op_) {
	this->_INs->push_back(cacu_op_);
	return this;
}

cacu_op* cacu_op::operator >>(cacu_op *&cacu_op_) {
	cacu_op_->_INs->push_back(this);
	return cacu_op_;
}

cacu_op* cacu_op::operator <<(blob_base *&data_){
	this->_op->in_datas()->push_back(data_);
	data_->_REC();
	return this;
}

}
