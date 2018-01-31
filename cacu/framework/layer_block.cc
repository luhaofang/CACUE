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

layer_block::layer_block(size_t output_channel, size_t kernel_size,
		size_t stride, size_t pad, size_t input_dim, size_t channel) {
	_output_dim = 0;
	_channel = channel;
	_input_dim = input_dim;
	_output_channel = channel;
	_kernel_size = kernel_size;
	_stride = stride;
	_pad = pad;

	_layers = new vector<layer_base*>();
}

layer_block::~layer_block() {
	for(int i = 0; i < _layers->size(); ++i)
	{
		delete _layers->at(i);
		_layers->at(i)= NULL;
	}
	delete _layers;
}

layer_block& layer_block::operator <<(layer_block* const &layer_block_) {
	for (int i = 0; i < layer_block_->length(); ++i)
		_layers->push_back(layer_block_->layer_bases(i));
	_output_dim = pop_layer()->_output_dim;
	return *this;
}

layer_block& layer_block::operator <<(layer_block &layer_block_) {
	for (int i = 0; i < layer_block_.length(); ++i)
		_layers->push_back(layer_block_.layer_bases(i));
	_output_dim = pop_layer()->_output_dim;
	return *this;
}

layer_block& layer_block::operator <<(layer_base* const &layer_) {

	_layers->push_back(layer_);
	_output_dim = layer_->_output_dim;
	return *this;
}

layer_block& layer_block::operator <<(layer_base &layer_) {

	_layers->push_back(&layer_);
	_output_dim = layer_._output_dim;
	return *this;
}

}
