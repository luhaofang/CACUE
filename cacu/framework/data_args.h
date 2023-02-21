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

#ifndef DATA_ARGS_H_
#define DATA_ARGS_H_

#include "args_base.h"

#include "../utils/log.h"

using namespace std;

namespace cacu {

class data_args: public args_base<int> {

public:

	data_args(int _output_channel, int _kernel_size, int _stride, int _pad, int _channel) :
			args_base<int>(_output_channel, _kernel_size, _stride, _pad, _channel, _ARGSEND) {

	}

	data_args() :
				args_base<int>(_ARGSEND) {

	}

	~data_args() {

	}

	inline int output_channel() const {
		return this->at(0);
	}

	inline int kernel_size() const {
		return this->at(1);
	}

	inline int stride() const {
		return this->at(2);
	}

	inline int pad() const {
		return this->at(3);
	}

	inline int channel() const {
		return this->at(4);
	}

	inline void serialize(std::ostream &os) {
		int size_ = this->size();
		os.write((char*)(&size_), sizeof(int));
		for(unsigned int i = 0; i < size(); ++i)
			os.write((char*)(&(this->at(i))), sizeof(int));
	}

	inline void load(std::istream &is) {
		if(size() != 0)
			LOG_FATAL("Data args needs to be an empty container!");
		int size = 0;
		is.read(reinterpret_cast<char*>(&size), sizeof(int));
		int d_size = 0;
		for(unsigned int i = 0; i < size; ++i){
			is.read(reinterpret_cast<char*>(&d_size), sizeof(int));
			this->push_back(d_size);
		}
	}

private:

};
}

#endif
