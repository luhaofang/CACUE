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

#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer_base.h"
//#include "layer.h"
#include "layer_block.h"

namespace cacu {

class network {

public:
	network(blobs *&datas_);

	~network();

	/*
	 * Class network unlike layer & layer_block,
	 * it maintains the abstract layer's operator entity.
	 *
	 */
	network& operator <<(layer_block* const &layer_block_);

	network& operator <<(layer_block &layer_block_);

	network& operator <<(layer_base* const &layer_);

	network& operator <<(layer_base &layer_);

	inline layer *&layers(int i) {
		return (layer*&) _layers->at(i);
	}

	inline layer_base *&layer_bases(int i) {
		return _layers->at(i);
	}

	inline int layer_count() const {
		return _layers->size();
	}

	void predict();

	void forward_propagate();

	void back_propagate();

	void set_weights_type(param_init_type type_, float_t value);

	inline void set_weight(int op_id, param_init_type type_, float_t value) {
		get_op(op_id)->get_weight(0)->set_init_type(type_, value);
	}

	inline operator_base *&get_op(int i) {
		return _ops->at(i);
	}

	inline int op_count() const {
		return _ops->size();
	}

	inline blobs *& input_blobs() {
		return _input_blobs;
	}

	inline blob *& output_blob() {
		return _ops->at(_ops->size() - 1)->out_data<blob>();
	}

	void output_blobs();

	inline phase_type phase() {
		return _input_blobs->at(0)->phase();
	}

	void load_weights(chars_t modelpath);

	void load_weights_from(chars_t modelpath, int op_num);

	void save_weights(chars_t modelpath);

	void check();

	void set_phase(phase_type phase_);

private:

	vector<layer_base*> *_layers;

	vector<operator_base*> *_ops;

	blobs* _input_blobs;

};

}

#endif