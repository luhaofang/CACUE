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

#ifndef PRUNEABLE_SUM_ELEMWISE_OP_H_
#define PRUNEABLE_SUM_ELEMWISE_OP_H_

#include "../ops/ops_definition.h"
#include "../ops/operator_base.h"


namespace cacu {

class pruneable_sum_elemwise_op: public operator_base {

public:

	pruneable_sum_elemwise_op(blobs *&data, op_args *& _o_args);

	~pruneable_sum_elemwise_op();

	void initial() override;

	void check() override;

	void init_weights() override;

	void op(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void grad(blobs *&s_blobs_,blobs *&o_blobs_) override;

	void load(std::ifstream& is) override;

	void save(std::ostream& os) override;

	void echo() override;

	void prune_blobs();

	//align with data blob
	inline void add_prune_channel(vec_i& pruned_channel){
		_pruned_channels->push_back(pruned_channel);
	}

	inline void _prepare_for_pruning(){
		_o_args->at(0) = ((float_t)_out_channels);
		_o_args->at(1) = ((float_t)_original_channels);
	}

	inline vec_i *get_pruned_vec() const {
		return _pruned_channel;
	}

	inline void set_load_param(bool load_) {
		_load_param = load_;
	}

private:

	vector<vec_i> *_pruned_channels;

	vec_i *_pruned_channel;

	bool _load_param = false;

	int _out_channels = 0;

	int _original_channels = 0;
};
}


#endif
