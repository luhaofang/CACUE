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
#include "../tensor/tensor.h"

using namespace std;

namespace cacu {

class blob: public blob_base {

public:

	blob(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
			float_t _value, phase_type phase, bool _malloc_when_init = true);

	~blob();

	void _RELEASE_BLOB() {
		_REC_ -= 1;
		if (_REC_ < 0) {
			delete this;
		}
	}
	/*
	 * initial the data memory.
	 * if using dynamic operator, the function will be called after computing graphic setup.
	 */
	void _MALLOC() {
		_tdata = new tensor<float_t>(count());
		_s_data = _tdata->pdata();
		_tdata->set_value(_init_value);
		if (train == _phase) {
			_tdiff = new tensor<float_t>(count());
			_s_diff = _tdiff->pdata();
			//_tdiff->set_value(_value);
		}
		_IS_MALLOCED_ = true;
	}

	/**
	 * return the piece probe in blob data
	 */
	inline float_t* p_data(dsize_t n) const {
		CHECK_LT_OP(n, num(), "Index out of range %d vs %d!", n, num() - 1);
		return (float_t*) _s_data + n * length();
	}

#if __USE_DEVICE__ == ON
	/**
	 * return the piece probe in blob data
	 */
	inline float_t* p_data_cpu(dsize_t n, bool sync = true) const {
		CHECK_LT_OP(n, num(), "Index out of range %d vs %d!", n, num() - 1);
		return _tdata->pdata_cpu(sync) + n * length();
	}

	/**
	 * return the source probe in blob data
	 */
	inline float_t* s_data_cpu(bool sync = true) const {
		return _tdata->pdata_cpu(sync);
	}

	/**
	 * return the piece probe in blob data
	 */
	inline float_t* p_diff_cpu(dsize_t n, bool sync = true) const {
		CHECK_LT_OP(n, num(), "Index out of range %d vs %d!", n, num() - 1);
		return _tdiff->pdata_cpu(sync) + n * length();
	}

	/**
	 * return the source probe in blob data
	 */
	inline float_t* s_diff_cpu(bool sync = true) const {
		return _tdiff->pdata_cpu(sync);
	}

#endif
	/**
	 * return the piece probe in blob diff
	 */
	inline float_t* p_diff(dsize_t n) const {
		CHECK_LT_OP(n, num(), "Index out of range %d vs %d!", n, num() - 1);
		return (float_t*) _s_diff + n * length();
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
	inline void set_data(float_t value_) {
		_tdata->set_value(value_);
	}

	/**
	 *
	 */
	inline void set_diff(float_t value_) {
		_tdiff->set_value(value_);
	}

	inline void set_data_from(int pIndex_, int length_, float_t value_) {
		CHECK_GE_OP(pIndex_, 0, "check pIndex >= 0 vs %d", pIndex_);
		CHECK_LT_OP(pIndex_, count(), "check pIndex < count vs %d", pIndex_);
		CHECK_LE_OP(pIndex_ + length_, count(), "pindex + length must <= count vs %d, %d", pIndex_, length_);
		_tdata->set_value(pIndex_, length_, value_);
	}

	inline void set_diff_from(int pIndex_, int length_, float_t value_) {
		CHECK_GE_OP(pIndex_, 0, "check pIndex >= 0 vs %d", pIndex_);
		CHECK_LT_OP(pIndex_, count(), "check pIndex < count vs %d", pIndex_);
		CHECK_LE_OP(pIndex_ + length_, count(), "pindex + length must <= count vs %d, %d", pIndex_, length_);
		_tdiff->set_value(pIndex_, length_, value_);
	}

	/**
	 *
	 */
	inline void set_pdata(float_t value_, int i) {
		_tdata->set_value(i * length(), length(), value_);
	}

	/**
	 *
	 */
	inline void set_pdiff(float_t value_, int i) {
		_tdiff->set_value(i * length(), length(), value_);
	}

	inline bool is_malloced(){
		return _IS_MALLOCED_;
	}

	/**
	 * copy dest blob data to local blob
	 */
	void copy_blob(const blob* blob_);

	/**
	 * copy dest blob data to local blob
	 */
	void copy2diff(const blob* blob_);

	void copy2data(const blob* blob_);

	blob* copy_create(phase_type phase_, float_t value_) const;

	/*
	 * copy data dsize_to blob, if blob is established in gpu, io op is needed
	 * where i is the start piece index in blob
	 */
	void copy2data(vec_t &data_, dsize_t i);

	void copy2data(vec_t &data_, dsize_t start, dsize_t length_);

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

	void load_from(const chars_t& path_) {
		std::ifstream is(path_, ios::binary);
		is.precision(numeric_limits<float_t>::digits10);
		if (!is)
			LOG_FATAL("file %s cannot be opened!", path_.c_str());
		string line = "";
		LOG_INFO("Load data from %s!", path_.c_str());
#if __USE_DEVICE__ == ON
		vec_t _v(count());

		int i = 0;
		while (getline(is, line)) {
			_v[i] = strtof(line.c_str(), NULL);
			i += 1;
		}
		device_copy2dev(s_data(), &_v[0], count());
#else
		int i= 0;
		while(getline(is, line))
		{
			s_data()[i] = strtof(line.c_str(), NULL);
			i+=1;
		}
#endif
		is.close();
	}

	void output_bin(const chars_t& path_);

	void input_bin(const chars_t& path_);

	inline dsize_t calculate_size() {
		return test == _phase ?
				count() * sizeof(float_t) : 2 * count() * sizeof(float_t);
	}

	inline void _RESET_DATA() {
		if (_variable) {
			if (_tdata != NULL)
				_tdata->refresh();
			if (_tdiff != NULL && train == _phase)
				_tdiff->refresh();
		}
	}

	inline void _RESET_DIFF() {
		if (_variable) {
			if (_tdiff != NULL && train == _phase)
				_tdiff->refresh();
		}
	}

	void floor();

	void clip(float_t lt, float_t rt);

	void serializa(std::ostream& os);

	void load(std::ifstream& is);

	void resize(dsize_t num, dsize_t channel, dsize_t width, dsize_t height);

	void set_init_type(param_init_type type, float_t value = 0);

	void switch_channel();

	void switch_channel_data();

	void switch_channel_data(const dsize_t num_, const dsize_t channel_, const dsize_t channel_length_);

	void switch_body();

	//type_code 0: rotate width and height with channel.
	//type_code 1: rotate width and height without channel
	void switch_random_rotate(int type_code);

	inline void init_blob_data(void init_blob(blob*)) {
		init_blob(this);
	}

#if __USE_CUDNN__ == ON
	inline cudnnTensorDescriptor_t tensor_desc() {
		return _data_desc;
	}

	inline void set_tensor_desc(int num_, int channel_, int width_,
			int height_) {
		set_tensor_4d_desc(_data_desc, num_, channel_, width_, height_);
	}

	inline void set_tensor_desc(int num_, int channel_, int width_, int height_,
			int stride_num_, int stride_channel_, int stride_width_,
			int stride_height_) {
		set_tensor_4d_desc(_data_desc, num_, channel_, width_, height_,
				stride_num_, stride_channel_, stride_width_, stride_height_);
	}

#endif

protected:

	tensor<float_t>* _tdata = NULL;

	tensor<float_t>* _tdiff = NULL;

	float_t _init_value;

	bool _IS_MALLOCED_;

private:

#if __USE_CUDNN__ == ON
	cudnnTensorDescriptor_t _data_desc = NULL;
#endif

};
}

#endif
