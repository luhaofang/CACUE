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

#ifndef LAYER_H_
#define LAYER_H_

#include "../ops/ops_definition.h"

#include "layer_base.h"

namespace cacu {

class layer: public layer_base {

public:

	layer(data_args *args);

	layer();

	~layer(){}

	int caculate_data_space() {
		return 0;
	}

	layer* op(op_name op_);

	layer* op(op_name op_, blob_base * blob_);

	layer* op(op_name op_, blob_base * blob_, data_args *& args_);

	layer* op(op_name op_, blobs * blobs_);

	layer* op(op_name op_, blobs * blobs_, data_args *& args_);

	layer* op(op_name op_, blob_base * blob_, op_args * args_);

	template<class OPTYPE>
	inline OPTYPE *& get_op(int i) const {
		return (OPTYPE*&) _ops->at(i);
	}

	inline operator_base *&get_head_op() {
		return _ops->at(0);
	}

	inline blob_base *&get_oblob() {
		return out_blob;
	}

	inline blobs *&get_oblobs() const {
		return _ops->back()->out_datas();
	}

private:


};

}


#endif