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

#ifndef POOLING_BASE_OP_H_
#define POOLING_BASE_OP_H_

namespace cacu {

class pooling_base_op: public operator_base {

public:

	pooling_base_op(blobs *&data, data_args *&args_, op_name type_) :
			operator_base(data, args_, type_) {
	}

	~pooling_base_op() {

	}

	void init_weights() override {
		return;
	}

	void check() override {
		if(_args == NULL)
			LOG_FATAL("pooling data args cannot equal to NULL!");
		//kernel_size > 0
		CHECK_GT_OP(_args->kernel_size(), 0, "kernel_size must > 0 vs %d",
				_args->kernel_size());
		//CHECK_EQ_OP(_args->output_channel(),s_blob->channel(),"source data must equal to layer args output_channel!");
	}

	void load(std::ifstream& is) override {
		return;
	}

	void save(std::ostream& os) override {
		return;
	}


};
}


#endif
