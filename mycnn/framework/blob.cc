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
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS size_tERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "blob.h"

#include "../mem/tensor.h"
#include "../mem/mem_functions.hpp"

using namespace std;

namespace mycnn {

blob::blob(size_t num, size_t channel, size_t width, size_t height,
		float_t _value, phase_type phase) :
		blob_base(num, channel, width, height, phase, __blob__) {
	_tdata = new tensor<float_t>(_length);
	if (train == phase) {
		_tdiff = new tensor<float_t>(_length);
	}
}

blob::~blob() {
	delete _tdata;
	if (train == _phase) {
		delete _tdiff;
	}
}

/**
 * reset all data (data & diff) in this blob
 */
inline const void blob::_RESET_DATA() {
	_tdata->refresh();
	if (train == _phase)
		_tdiff->refresh();
}

/**
 * reset diff data (diff) in this blob
 */
inline const void blob::_RESET_DIFF() {
	_tdiff->refresh();
}


inline size_t blob::calculate_size() {
	return test == _phase ?
			_length * sizeof(float_t) : 2 * _length * sizeof(float_t);
}

/**
 * copy dest blob data to local blob
 */
inline void blob::copy_blob(blob* blob_) {
	CHECK_EQ_OP(blob_->count(), _length, "blob size must be equal! %d vs %d",
			blob_->count(), _length);
	cacu_copy(blob_->s_data(), _length, (float_t*) _s_data);
	CHECK_EQ_OP(blob_->phase(), _phase, "blob phase must be the same! %d vs %d",
			blob_->phase(), _phase);
	if (train == _phase && train == blob_->phase()) {
		cacu_copy(blob_->s_diff(), _length, (float_t*) _s_diff);
	}
}

inline blob* blob::copy_create(phase_type phase_, float_t value_) const {
	return NULL;
	//return new blob(_num, _channel, _width, _height, value_, phase_);
}

/*
 * copy data size_to blob, if blob is established in gpu, io op is needed
 * where i is the start index in blob
 */
inline void blob::copy2data(vec_t &data_, size_t i) {
	CHECK_EQ_OP(data_.size(), _cube_length, "blob size must be equal! %d vs %d",
			data_.size(), _cube_length);
	_tdata->copy2data(i*_cube_length, _cube_length, &data_[0]);
}

/*
 * copy data size_to blob, if blob is established in gpu, io op is needed
 */
inline void blob::copy2data(vec_t &data_) {
	CHECK_EQ_OP(data_.size(), _length, "blob size must be equal! %d vs %d",
			data_.size(), _length);
	_tdata->copy2data(&data_[0]);
}

/*
 * copy data size_to blob, if blob is established in gpu, io op is needed
 * where i is the start index in blob
 */
inline void blob::copy2diff(vec_t &data_, size_t i) {
	CHECK_EQ_OP(data_.size(), _cube_length, "blob size must be equal! %d vs %d",
			data_.size(), _cube_length);
	_tdiff->copy2data(i*_cube_length, _cube_length, &data_[0]);
}

/*
 * copy data size_to blob's diff, if blob is established in gpu, io op is needed
 */
inline void blob::copy2diff(vec_t &data_) {
	CHECK_EQ_OP(data_.size(), _length, "blob size must be equal! %d vs %d",
			data_.size(), _length);
	_tdiff->copy2data(&data_[0]);
}

/*
 * serializa blob data, output data to model file
 */
inline const void blob::serializa(std::ostream& os) {
	float_t* s_data_ = (float_t*) _s_data;
#if __USE_DEVICE__ == ON
	os.write((char*) (&_length), sizeof(_length));
	vec_t _v(_length);
	device_copy2host(&_v[0], (float_t*) _s_data, _length);
	for (int i = 0 ; i < _v.size(); ++i)
		os.write((char*) (&_v[i]), sizeof(float_t));
	vec_t().swap(_v);
#else
	os.write((char*)(&_length), sizeof(_length));
	for(size_t i = 0; i < _length; ++i)
	os.write((char*)(&s_data_[i]), sizeof(s_data_[i]));
#endif
}

/*
 * loads blob data from model file
 */
inline const void blob::load(std::ifstream& is) {
	float_t* s_data_ = (float_t*) _s_data;
#if __USE_DEVICE__ == ON
	vec_t _v(_length);
	size_t length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(size_t));
	CHECK_EQ_OP(length_, _length,
			"parameter length is not equal to local weight: %d vs %d!", length_,
			_length);
	for (size_t i = 0; i < length_; i++) {
		is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
	}
	device_copy2dev((float_t*) _s_data, &_v[0], length_);
	vec_t().swap(_v);
#else
	size_t length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(size_t));
	CHECK_EQ_OP(length_,_length,"parameter length is not equal to local weight: %d vs %d!",length_,_length);
	for (size_t i = 0; i < length_; i++) {
		is.read(reinterpret_cast<char*>(s_data_ + i), sizeof(float_t));
	}
#endif
}

inline const void blob::resize(size_t num, size_t channel, size_t width,
		size_t height) {
	_width = width;
	_height = height;
	_channel = channel;
	_num = num;
	_channel_length = width * height;
	_cube_length = channel * width * height;
	_length = _num * _cube_length;


}

}
