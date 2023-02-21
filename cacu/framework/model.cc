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


#include "model.h"

namespace cacu {

model::model()
{
	_cacu_ops = new map<chars_t, cacu_op*>();

	_params = new vector<weight *>();
}

model::~model(){
	map<chars_t, cacu_op*>::iterator iter = _cacu_ops->begin();
	while(iter != _cacu_ops->end())
	{
		delete iter->second;
		iter->second = NULL;
		_cacu_ops->erase(iter);
		iter++;
	}
	delete _cacu_ops;
	_cacu_ops = NULL;

	delete _params;
	_params = NULL;
}

void model::nn(chars_t op_name_, cacu_op *&op_)
{
	int nRet = _cacu_ops->count(op_name_);
	if(nRet > 0){
		LOG_WARNING("op [\'%s\'] is already defined!",op_name_.c_str());
		return;
	}
	this->_cacu_ops->insert(pair<chars_t, cacu_op*&>(op_name_, op_));
	for(int i = 0; i < op_->get_weights()->size(); ++i)
	{
		_params->push_back(op_->get_param(i));
	}
}





}
