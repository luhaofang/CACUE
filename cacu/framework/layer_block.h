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

#ifndef LAYER_BLOCK_H_
#define LAYER_BLOCK_H_

#include <vector>

#include "layer.h"

using namespace std;

namespace cacu {

class layer_block {

public:

	layer_block();

	layer_block(blobs *&datas_);

	~layer_block();

	inline blob_base * get_oblob() {
		assert(length() > 0);
		return layers(length() - 1)->get_oblob();
	}

	inline blob *oblob(){
		return dynamic_cast<blob*>(get_oblob());
	}

	template<typename OPTYPE>
	inline OPTYPE *top_op(){
		assert(length() > 0);
		layer *top_layer_=layers(length() - 1);
		OPTYPE* op = dynamic_cast<OPTYPE*>(top_layer_->get_out_op());
		if(op == NULL)
			LOG_FATAL("op type casting failed!");
		return op;
	}

	inline layer *layers(int i) {
		assert(i >= 0);
		layer* layer_ = dynamic_cast<layer*>(_layers->at(i));
		if(layer_ == NULL)
			LOG_FATAL("layer capture falled!");
		return layer_;
	}

	inline layer* pop_layer() const {
		assert(length() > 0);
		return _layers->at(length() - 1);
	}

	inline layer* layer_bases(int i) const {
		assert(i >= 0);
		return _layers->at(i);
	}

	inline int length() const {
		return _layers->size();
	}

	layer_block& operator <<(layer_block* const &layer_block_);

	layer_block& operator <<(layer_block &layer_block_);

	layer_block& operator <<(layer* const &layer_);

	layer_block& operator <<(layer &layer_);

	void load_weights(const chars_t& modelpath){
		std::ifstream is(modelpath, ios::binary);
		is.precision(std::numeric_limits<float_t>::digits10);
		if (!is)
			LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
		for(unsigned int i =0 ; i < _layers->size(); ++i)
		{
			_layers->at(i)->load_weights(is);
		}
		is.close();
		LOG_INFO("Initialize model by : %s", modelpath.c_str());
	}


	void load_weights_reverse(const chars_t& modelpath){
		std::ifstream is(modelpath, ios::binary);
		is.precision(std::numeric_limits<float_t>::digits10);
		if (!is)
			LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
		for(unsigned int i = _layers->size() - 1 ; i >=0; --i)
		{
			_layers->at(i)->load_weights_reverse(is);
		}
		is.close();
		LOG_INFO("Initialize model by : %s", modelpath.c_str());
	}

	void save_weights(const chars_t& modelpath){
		std::ofstream os(modelpath, ios::binary);
		os.precision(std::numeric_limits<float_t>::digits10);
		if (!os)
			LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
		for(unsigned int i =0 ; i < _layers->size(); ++i)
		{
			_layers->at(i)->save_weights(os);
		}
		os.close();
		LOG_INFO("The model is saved at : %s", modelpath.c_str());
	}

	void set_update_weight(bool isupdate_);

	inline blobs * get_indatas(){
		return _in_datas;
	}


private:

	vector<layer*> *_layers;

	blobs *_in_datas;
};

}

#endif
