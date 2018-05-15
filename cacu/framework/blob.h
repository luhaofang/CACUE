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

#ifndef BLOB_H_
#define BLOB_H_

#include <limits>

#include "blob_base.h"

//#include "../definition.h"

#include "../tensor/tensor.h"

using namespace std;

namespace cacu {

class blob: public blob_base {

public:

	blob(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
			float_t _value, phase_type phase);

	~blob();

	/**
	 * return the piece probe in blob data
	 */
	inline float_t* p_data(dsize_t n) const {
		//CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num - 1);
		return (float_t*) _s_data + n * _cube_length;
	}

#if __USE_DEVICE__ == ON
	/**
	 * return the piece probe in blob data
	 */
	inline float_t* p_data_cpu(dsize_t n) const {
		//CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num - 1);
		return _tdata->pdata_cpu() + n * _cube_length;
	}

	/**
	 * return the source probe in blob data
	 */
	inline float_t* s_data_cpu() const {
		return _tdata->pdata_cpu();
	}

#endif
	/**
	 * return the piece probe in blob diff
	 */
	inline float_t* p_diff(dsize_t n) const {
		//CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num - 1);
		return (float_t*) _s_diff + n * _cube_length;
	}

	/**
	 * return the source probe in blob data
	 */
	inline float_t* s_data() const {
		return (float_t*) _s_data;
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
	inline void set_data(float_t value_)
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
	void copy_blob(blob* blob_);

	blob* copy_create(phase_type phase_, float_t value_) const;

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 * where i is the start piece index in blob
	 */
	void copy2data(vec_t &data_, dsize_t i);

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 * where i is the start piece index in blob
	 */
	void copy2data(float_t *data_, dsize_t i);

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 */
	void copy2data(vec_t &data_);

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 * where i is the start piece index in blob
	 */
	void copy2diff(vec_t &data_, dsize_t i);

	/*
	 * copy data dsize_to blob's diff, if blob is established in gpu, io op is needed
	 */
	void copy2diff(vec_t &data_);

	void load_from(chars_t path_);

	void output_bin(chars_t path_);

	void input_bin(chars_t path_, int i);

	inline dsize_t calculate_size() {
		return test == _phase ?
				_length * sizeof(float_t) : 2 * _length * sizeof(float_t);
	}

	inline void _RESET_DATA() {
		_tdata->refresh();
		if (train == _phase)
			_tdiff->refresh();
	}

	inline void _RESET_DIFF() {
		if (train == _phase)
			_tdiff->refresh();
	}

	void serializa(std::ostream& os);

	void load(std::ifstream& is);

	inline void resize(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height) {
		_width = width;
		_height = height;
		_channel = channel;
		_num = num;
		_channel_length = width * height;
		_cube_length = channel * width * height;
		_length = _num * _cube_length;

		if (_IS_MOTIFIED())
			return;
		_tdata->resize(_length, 0);
		_s_data = _tdata->pdata();
		if (_tdiff != NULL) {
			_tdiff->resize(_length, 0);
			_s_diff = _tdiff->pdata();
		}
	}

	inline void resize(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, float_t value) {

		_width = width;
		_height = height;
		_channel = channel;
		_num = num;
		_channel_length = width * height;
		_cube_length = channel * width * height;
		_length = _num * _cube_length;

		if (_IS_MOTIFIED())
			return;
		_tdata->resize(_length, value);
		_s_data = _tdata->pdata();
		if (_tdiff != NULL) {
			_tdiff->resize(_length, value);
			_s_diff = _tdiff->pdata();
		}
	}

	void set_init_type(param_init_type type, float_t value);

protected:

	tensor<float_t>* _tdata = NULL;

	tensor<float_t>* _tdiff = NULL;

};
}


#endif
