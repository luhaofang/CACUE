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

#include "../mycnn.h"

namespace mycnn{

class predictor
{
public:
	predictor(network *&net_){
		_net = net_;
	};

	~predictor(){};

#if __PARALLELTYPE__ == __GPU__

	/**
	 * initial state of the gpu device
	 * if gpu running is need, cacu create initialize the gpu states for predictor
	 */
	void initial(int device_id = 0){
		if(cuda_initial()){
			if(cudaSetDevice(device_id) == cudaErrorInvalidDevice){
				LOG_FATAL("Set Device %d occurred error",device_id);
			}
			else
			{
				cuda_set_device(device_id);
			}
		}
	}

#endif

	void predict(){
		_net->predict();
	}

	network *net_(){
		return _net;
	}
	/**
	 * output blob of the net's top layer
	 */
	blob *output(){
		return _net->output_blob();
	}

	/**
	 * if the network is flexible and fixed for multi-outputs, cacu will present the ends outputs of the network;
	 */
	blobs *outputs(){
		blobs* blobs_ = cacu_allocator::create_blobs();
		for(int i = 0 ; i < _net->op_count();++i)
		{
			operator_base *op = _net->get_op(i);
			if(op->out_data()->is_output())
				blobs_->push_back(op->out_data());
		}
		return blobs_;
	}

private:

	network *_net;


};
};

