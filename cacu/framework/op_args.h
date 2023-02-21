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

#ifndef OP_ARGS_H_
#define OP_ARGS_H_

#include <vector>

#include "../utils/log.h"
#include "../utils/check_utils.h"

using namespace std;

namespace cacu {

class op_args : public vector<float_t>{

public:

	op_args() : vector<float_t>(){

	}

	op_args(float_t args_) : vector<float_t>(){
		this->push_back(args_);
	}

	op_args(float_t args1_,float_t args2_) : vector<float_t>(){
		this->push_back(args1_);
		this->push_back(args2_);
	}

	op_args(float_t args1_,float_t args2_,float_t args3_) : vector<float_t>(){
		this->push_back(args1_);
		this->push_back(args2_);
		this->push_back(args3_);
	}

	op_args(float_t args1_,float_t args2_,float_t args3_,float_t args4_)  : vector<float_t>(){
		this->push_back(args1_);
		this->push_back(args2_);
		this->push_back(args3_);
		this->push_back(args4_);
	}

	~op_args() {

	}

	inline void serialize(std::ostream &os) {

		int size_ = this->size();
		os.write((char*)(&size_), sizeof(int));
		for(unsigned int i = 0; i < size(); ++i)
			os.write((char*)(&(this->at(i))), sizeof(float_t));
	}

	inline void load(std::istream &is) {
		if(size() != 0)
			LOG_FATAL("Op args needs to be an empty container!");
		int size = 0;
		is.read(reinterpret_cast<char*>(&size), sizeof(int));
		float_t d_size = 0;
		for(unsigned int i = 0; i < size; ++i){
			is.read(reinterpret_cast<char*>(&d_size), sizeof(float_t));
			this->push_back(d_size);
		}
	}

};

}

#endif
