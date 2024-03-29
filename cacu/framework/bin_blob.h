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


#ifndef BIN_BLOB_H_
#define BIN_BLOB_H_

#include <bitset>

#include "blob_base.h"

#include "../tensor/tensor.h"

using namespace std;

namespace cacu{

class bin_blob : public blob_base {

public:

	bin_blob(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
			int _value, phase_type phase);

	~bin_blob();

	void _RELEASE_BIN_BLOB()
	{
		_REC_ -= 1;
		if(_REC_ < 0){
			delete this;
		}
	}

	/**
	 * return the piece probe in blob data
	 */
	inline int* p_data(dsize_t n) const {
		CHECK_LT_OP(n, num(), "Index out of range %d vs %d!", n, num() - 1);
		return (int*) _s_data + n * length();
	}

	/**
	 * return the piece probe in blob diff
	 */
	inline float_t* p_diff(dsize_t n) const {
		CHECK_LT_OP(n, num(), "Index out of range %d vs %d!", n, num() - 1);
		return (float_t*) _s_diff + n * length();
	}

#if __USE_DEVICE__ == ON
	/**
	 * return the piece probe in blob data
	 */
	inline int* p_data_cpu(dsize_t n, bool sync = true) const {
		CHECK_LT_OP(n ,num(), "Index out of range %d vs %d!",n ,num() - 1);
		return _tdata->pdata_cpu(sync) + n * length();
	}

	/**
	 * return the source probe in blob data
	 */
	inline int* s_data_cpu(bool sync = true) const {
		return _tdata->pdata_cpu(sync);
	}

#endif

	/**
	 * return the source probe in blob data
	 */
	inline int* s_data() const {
		return (int*) _s_data;
	}

	/**
	 * return the source probe in blob diff
	 */
	inline float_t* s_diff() const {
		return (float_t*) _s_diff;
	}

	/**
	 *
	 */
	inline void set_data(int value_)
	{
		_tdata->set_value(value_);
	}

	/**
	 *
	 */
	inline void set_diff(float_t value_)
	{
		_tdiff->set_value(value_);
	}

	/**
	 * copy dest blob data to local blob
	 */
	void copy_blob(const bin_blob* blob_);

	bin_blob* copy_create(phase_type phase_, int value_) const;

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 * where i is the start piece index in blob
	 */
	void copy2data(vec_i &data_, dsize_t i);

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 */
	void copy2data(vec_i &data_);

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 * where i is the start piece index in blob
	 */
	void copy2diff(vec_t &data_, dsize_t i);

	/*
	 * copy data dsize_to blob's diff, if blob is established in gpu, io op is needed
	 */
	void copy2diff(vec_t &data_);

	inline dsize_t calculate_size() {
		return test == _phase ?
				count() * sizeof(int) :
				count() * sizeof(int) + count() * sizeof(float_t);
	}

	inline void _RESET_DATA() {
		if(_variable){
			_tdata->refresh();
			if (_tdiff != NULL && train == _phase)
				_tdiff->refresh();
		}
	}

	inline void _RESET_DIFF() {
		if(_variable){
			if (_tdiff != NULL && train == _phase)
				_tdiff->refresh();
		}
	}

	void serializa(std::ostream& os);

	void load(std::ifstream& is);

	void resize(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height) {
		_body->set_body(num,channel,width,height);

		if(_IS_MOTIFIED())
			return;
		if(_tdata != NULL){
			_tdata->resize(count(), 0);
			_s_data = _tdata->pdata();
		}
		if(_tdiff != NULL){
			_tdiff->resize(count(), 0);
			_s_diff = _tdiff->pdata();
		}
	}

	void flip();

private:

	tensor<int>* _tdata = NULL;

	tensor<float_t>* _tdiff = NULL;

};

}


#endif
