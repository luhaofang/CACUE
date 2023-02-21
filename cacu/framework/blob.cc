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

#include <string>
#include <stdlib.h>
#include "blob.h"

#include "../math/math_functions.h"
#include "../math/utils/rand_t.h"
#include "../utils/data_printer.h"

using namespace std;

namespace cacu {

blob::blob(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
		float_t _value, phase_type phase, bool _malloc_when_init) :
		blob_base(num, channel, width, height, phase, __blob__) {
	_init_value = _value;
#if __USE_CUDNN__ == ON
	create_tensor_4d_desc(_data_desc);
	if (_body->_length != 0) {
		set_tensor_4d_desc(_data_desc, _body->_num, _body->_channel,
				_body->_width, _body->_height);
	} else {
		LOG_WARNING("Create a blob with size of 0!");
	}
#endif
	if (_malloc_when_init)
		_MALLOC();
	_IS_MALLOCED_ = _malloc_when_init;
}

blob::~blob() {

	delete _tdata;
	_tdata = NULL;
	if (_tdiff != NULL) {
		delete _tdiff;
		_tdiff = NULL;
	}
#if __USE_CUDNN__ == ON
	destroy_tensor_descriptor(_data_desc);
#endif
}

/**
 * copy dest blob data to local blob
 */
void blob::copy_blob(const blob* blob_) {
	this->_CHECK_SIZE_EQ(blob_);
	_tdata->copy2data(blob_->s_data());
	CHECK_EQ_OP(blob_->phase(), _phase, "blob phase must be the same! %d vs %d",
			blob_->phase(), _phase);
	if (train == _phase && train == blob_->phase()) {
		//_tdiff->copy2data(blob_->s_diff());
		_tdiff->set_value(0);
	}
}

void blob::copy2diff(const blob* blob_) {
	this->_CHECK_SIZE_EQ(blob_);
	assert(blob_->phase() == train);
	CHECK_EQ_OP(blob_->phase(), _phase, "blob phase must be the same! %d vs %d",
			blob_->phase(), _phase);
	_tdiff->copy2data(blob_->s_diff());
}

void blob::copy2data(const blob* blob_) {
	this->_CHECK_SIZE_EQ(blob_);
	_tdata->copy2data(blob_->s_data());
}

blob* blob::copy_create(phase_type phase_, float_t value_) const {
	return new blob(num(), channel(), width(), height(), value_, phase_);
}

/*
 * copy data dsize_to blob, if blob is established in gpu, io op is needed
 * where i is the start index in blob
 */
void blob::copy2data(vec_t &data_, dsize_t i) {
	CHECK_EQ_OP((int )data_.size(), length(),
			"blob size must be equal! %d vs %d", (int )data_.size(), length());
	_tdata->copy2data(i * length(), length(), &data_[0]);
}

void blob::copy2data(vec_t &data_, dsize_t start, dsize_t length_) {

	CHECK_EQ_OP((int )data_.size(), length_,
			"blob size must be equal! %d vs %d", (int )data_.size(), length_);
	_tdata->copy2data(start, length_, &data_[0]);
}

void blob::copy2data(float_t *data_, dsize_t i) {
	_tdata->copy2data(i * length(), length(), data_);
}

/*
 * copy data dsize_to blob, if blob is established in gpu, io op is needed
 */
void blob::copy2data(vec_t &data_) {
	CHECK_EQ_OP((int )data_.size(), count(),
			"blob size must be equal! %d vs %d", (int )data_.size(), count());
	_tdata->copy2data(&data_[0]);
}

/*
 * copy data dsize_to blob, if blob is established in gpu, io op is needed
 * where i is the start index in blob
 */
void blob::copy2diff(vec_t &data_, dsize_t i) {
	CHECK_EQ_OP((int )data_.size(), length(),
			"blob size must be equal! %d vs %d", (int )data_.size(), length());
	_tdiff->copy2data(i * length(), length(), &data_[0]);
}

/*
 * copy data dsize_to blob's diff, if blob is established in gpu, io op is needed
 */
void blob::copy2diff(vec_t &data_) {
	CHECK_EQ_OP((int )data_.size(), count(),
			"blob size must be equal! %d vs %d", (int )data_.size(), count());
	_tdiff->copy2data(&data_[0]);
}

/*
 * serializa blob data, output data to model file
 */
void blob::serializa(std::ostream& os) {
	_tdata->serializa(os);
}

void blob::floor() {
	cacu_floor(s_data(), count(), s_data());
}

void blob::clip(float_t lt, float_t rt) {
	cacu_clip_vec(s_data(), lt, rt, count());
}

/*
 * loads blob data from model file
 */
void blob::load(std::ifstream& is) {
	dsize_t length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(dsize_t));
	CHECK_EQ_OP(length_, count(),
			"Parameter length is not equal to local length: %d vs %d!", length_,
			count());
	_tdata->load(is);
}

void blob::output_bin(const chars_t& path_) {
	std::ofstream os(path_, ios::binary);
	os.precision(numeric_limits<float_t>::digits10);
	if (!os)
		LOG_FATAL("file %s cannot be opened!", path_.c_str());
	_tdata->serializa(os);
	os.close();
}

void blob::input_bin(const chars_t& path_) {
	std::ifstream is(path_, ios::binary);
	is.precision(numeric_limits<float_t>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", path_.c_str());
	dsize_t length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(dsize_t));
	CHECK_EQ_OP(length_, count(),
			"Parameter length is not equal to local length: %d vs %d!", length_,
			count());
	_tdata->load(is);
	is.close();
}


void blob::set_init_type(param_init_type type, float_t value) {
	vec_t w(count());
	float_t d_value;
	switch (type) {
	case constant:
		for (int i = 0; i < count(); ++i)
			w[i] = value;
		break;
	case xavier:
		d_value = sqrtf(
				(float_t) 6.0 / (count() / num() + count() / channel()));
		if (value == 0.0)
			for (int i = 0; i < count(); ++i)
				w[i] = urand(-d_value, d_value);
		else
			for (int i = 0; i < count(); ++i)
				w[i] = urand(-d_value, d_value) * value;
		break;
	case gaussian:
		for (int i = 0; i < count(); ++i) {
			w[i] = gaussrand(value);
		}
		break;
	case msra:
		d_value = sqrtf((float_t) 2.0 / (channel() * height() * width()));
		if (value == 0.0)
			for (int i = 0; i < count(); ++i)
				w[i] = gaussrand(d_value);
		else
			for (int i = 0; i < count(); ++i)
				w[i] = gaussrand(d_value) * value;
		break;
	case uniform:
		for (int i = 0; i < count(); ++i)
			w[i] = urand(-value, value);
		break;
	default:
		LOG_FATAL("unknown initial type!");
		break;
	}
	_tdata->copy2data(&w[0]);
	vec_t().swap(w);
}

void blob::switch_channel() {
	cacu_transpose(s_data(), num(), channel(), channel_length());
	if (_phase == train)
		cacu_transpose(s_diff(), num(), channel(), channel_length());
	int temp_ = num();
	_body->_num = channel();
	_body->_channel = temp_;
	_body->_cube_length = _body->_channel * _body->_channel_length;
#if __USE_CUDNN__ == ON
	if (_body->_length != 0) {
		set_tensor_4d_desc(_data_desc, _body->_num, _body->_channel,
				_body->_width, _body->_height);
	} else {
		LOG_WARNING("Create a blob with size of 0!");
	}
#endif
}

void blob::switch_channel_data() {
	cacu_transpose(s_data(), num(), channel(), channel_length());
	if (_phase == train)
		cacu_transpose(s_diff(), num(), channel(), channel_length());
}

void blob::switch_channel_data(dsize_t num_, dsize_t channel_,
		dsize_t channel_length_) {
	cacu_transpose(s_data(), num_, channel_, channel_length_);
	if (_phase == train)
		cacu_transpose(s_diff(), num_, channel_, channel_length_);
}

void blob::switch_body() {
	int temp_ = num();
	_body->_num = channel();
	_body->_channel = temp_;
	_body->_cube_length = _body->_channel * _body->_channel_length;
#if __USE_CUDNN__ == ON
	if (_body->_length != 0) {
		set_tensor_4d_desc(_data_desc, _body->_num, _body->_channel,
				_body->_width, _body->_height);
	} else {
		LOG_WARNING("Create a blob with size of 0!");
	}
#endif
}

void blob::resize(dsize_t num, dsize_t channel, dsize_t width, dsize_t height) {
	_body->set_body(num, channel, width, height);
	if (_IS_MOTIFIED())
		return;
	if (_tdata != NULL) {
		_tdata->resize(count(), 0);
		_s_data = _tdata->pdata();
	}
	if (_tdiff != NULL) {
		_tdiff->resize(count(), 0);
		_s_diff = _tdiff->pdata();
	}
#if __USE_CUDNN__ == ON
	if (_body->_length != 0) {
		set_tensor_4d_desc(_data_desc, _body->_num, _body->_channel,
				_body->_width, _body->_height);
	} else {
		LOG_WARNING("Create a blob with size of 0!");
	}
#endif
}

void blob::switch_random_rotate(int type_code) {
	if (width() * height() == 1)
		return;
	switch (type_code) {
	case 0:
		for (int n = 0; n < num(); ++n) {
			int code_ = randint(10) > 5 ? randint(4) : 0;
			cacu_rotate(p_data(n), width(), height(), channel(),
					static_cast<rotate_code>(code_));
			int temp_ = width();
			_body->_width = _body->_height;
			_body->_height = temp_;
		}
		break;
	case 1:
		for (int n = 0; n < num(); ++n)
			for (int c = 0; c < channel(); ++c) {
				int code_ = randint(10) > 5 ? randint(4) : 0;
				cacu_rotate(p_data(n) + c * channel_length(), width(), height(),
						1, static_cast<rotate_code>(code_));
				int temp_ = width();
				_body->_width = _body->_height;
				_body->_height = temp_;
			}
		break;
	default:
		break;
	}
}

}
