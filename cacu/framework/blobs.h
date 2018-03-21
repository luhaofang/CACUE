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

#ifndef BLOBS_H_
#define BLOBS_H_

#include <vector>

#include "blob_base.h"
#include "blob.h"
#include "bin_blob.h"

using namespace std;

namespace cacu {

class blobs: public vector<blob_base*> {

public:

	blobs() {

	}

	~blobs() {
		for (unsigned i = 0; i < size(); ++i) {
			switch (at(i)->_TYPE()) {
			case __blob__:
				delete (blob*) at(i);
				at(i) = NULL;
				break;
			case __bin_blob__:
				delete (bin_blob*) at(i);
				at(i) = NULL;
				break;
				/*
				 case __em_blob__:
				 delete (em_blob*) at(i);
				 break;
				 case __em_bin_blob__:
				 delete (em_bin_blob*) at(i);
				 break;
				 */
			default:
				LOG_FATAL("can't identify the type!");
				break;
			}
		}
	}

	inline blobs& operator <<(blob_base* blob_base_) {
		this->push_back(blob_base_);
		return *this;
	}

	inline void _DELETE_BLOBS() {
		for (unsigned i = 0; i < size(); ++i) {
			switch (at(i)->_TYPE()) {
			case __blob__:
				delete (blob*) at(i);
				break;
			case __bin_blob__:
				delete (bin_blob*) at(i);
				break;
				/*
				 case __em_blob__:
				 delete (em_blob*) at(i);
				 break;
				 case __em_bin_blob__:
				 delete (em_bin_blob*) at(i);
				 break;
				 */
			default:
				LOG_FATAL("can't identify the type!");
				break;
			}
		}
		this->clear();
	}

	inline void _REC() {
		for (size_t i = 0; i < this->size(); ++i)
			at(i)->_REC();
	}

	/**
	 * reset all data (data & diff) in this blobs
	 */
	inline void _RESET_DATA() {
		for (size_t i = 0; i < this->size(); ++i)
			at(i)->_RESET_DATA();
	}

	/**
	 * reset diff data (diff) in this blobs
	 */
	inline void _RESET_DIFF() {
		for (size_t i = 0; i < this->size(); ++i)
			at(i)->_RESET_DIFF();
	}

	inline void _MOTIFY() {
		for (size_t i = 0; i < this->size(); ++i)
			at(i)->_MOTIFY();
	}

};
}


#endif