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

#include <vector>

#include "layer_base.h"
#include "layer.h"

using namespace std;

namespace cacu {

class layer_block {

public:

	layer_block();

	~layer_block();

	inline blob_base * get_oblob() {
		if (_layers->size() != 0)
			return layers(length() - 1)->get_oblob();
		return NULL;
	}

	inline layer* layers(int i) const {
		return (layer*) _layers->at(i);
	}

	inline layer_base*& pop_layer() const {
		return _layers->at(length() - 1);
	}

	inline layer_base*& layer_bases(int i) const {
		return _layers->at(i);
	}

	inline int length() const {
		return _layers->size();
	}

	layer_block& operator <<(layer_block* const &layer_block_);

	layer_block& operator <<(layer_block &layer_block_);

	layer_block& operator <<(layer_base* const &layer_);

	layer_block& operator <<(layer_base &layer_);

private:

	vector<layer_base*> *_layers;
};

}
