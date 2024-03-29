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

#include "layer_block.h"

namespace cacu {

layer_block::layer_block() {

	_layers = new vector<layer*>();
	_in_datas = NULL;
}

layer_block::layer_block(blobs *&datas_){
	_layers = new vector<layer*>();
	_in_datas = datas_;
}

layer_block::~layer_block() {
	/*
	for(size_t i = 0; i < _layers->size(); ++i)
	{
		delete _layers->at(i);
		_layers->at(i)= NULL;
	}
	*/
	//delete _layers;
	_layers = NULL;
	//make sure layer block DO NOT maintain data blob
	if(_in_datas != NULL)
		_in_datas = NULL;
}

layer_block& layer_block::operator <<(layer_block* const &layer_block_) {
	for (int i = 0; i < layer_block_->length(); ++i)
		_layers->push_back(layer_block_->layer_bases(i));
	return *this;
}

layer_block& layer_block::operator <<(layer_block &layer_block_) {
	for (int i = 0; i < layer_block_.length(); ++i)
		_layers->push_back(layer_block_.layer_bases(i));
	return *this;
}

layer_block& layer_block::operator <<(layer* const &layer_) {

	_layers->push_back(layer_);
	return *this;
}

layer_block& layer_block::operator <<(layer &layer_) {

	_layers->push_back(&layer_);
	return *this;
}

void layer_block::set_update_weight(bool isupdate_)
{
	for (unsigned int i = 0; i < _layers->size(); ++i) {
		for (int j = 0; j < _layers->at(i)->op_count(); ++j){
			_layers->at(i)->get_op_base(j)->set_is_update_weight(isupdate_);
			if(_layers->at(i)->get_op_base(j)->_TYPE()==CACU_BATCH_NORMALIZE)
			{
				dynamic_cast<batch_normalize_op*>(_layers->at(i)->get_op_base(j))->set_is_use_global_stats(isupdate_==false);
			}
		}
	}
}

}
