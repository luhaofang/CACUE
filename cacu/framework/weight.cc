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

#include "weight.h"

#include "../utils/log.h"
#include "../utils/check_utils.h"
#include "../math/utils/rand_t.h"

namespace cacu {

weight::weight(chars_t name, dsize_t num, dsize_t channel, dsize_t width,
		dsize_t height, phase_type phase) :
		blob(num, channel, width, height, 0, phase) {
	_name = name;
	_update_lr = 1.0;
	_decay_mult = 1.0;
	_update = true;
}

weight::~weight() {
}

void weight::set_init_type(param_init_type type, float_t value) {
	vec_t w(_length);
	switch (type) {
	case constant:
		for (int i = 0; i < _length; ++i)
			w[i] = value;
		break;
	case xavier:
		value = sqrt((float_t) 3.0 / (channel() * height() * width()));
		for (int i = 0; i < _length; ++i)
			w[i] = urand(-value, value);
		break;
	case gaussian:
		for (int i = 0; i < _length; ++i)
			w[i] = gaussrand(value);
		break;
	case msra:
		value = sqrt((float_t) 2.0 / (channel() * height() * width()));
		for (int i = 0; i < _length; ++i)
			w[i] = gaussrand(value);
		break;
	default:
		LOG_FATAL("unknown weight type for [%s]!", _name.c_str());
		break;
	}
	_tdata->copy2data(&w[0]);
	//vec_t().swap(w);
}

/*
 * serializa blob data, output data to model file
 */
void weight::serializa_group(std::ostream& os, int group) {
	float_t* p_data_;
	int length = _length / group;
#if __USE_DEVICE__ == ON
	os.write((char*) (&length), sizeof(length));
	vec_t _v(_channel_length * (_channel / group));
	for (int n = 0; n < _num; ++n) {
		p_data_ = s_data() + n * _v.size();
		device_copy2host(&_v[0], p_data_, _v.size());
		for (auto w : _v)
			os.write((char*) (&w), sizeof(w));
	}
	//vec_t().swap(_v);
#else

	os.write((char*)(&length), sizeof(length));
	vec_t _v(_channel_length * (_channel / group));
	for(int n = 0; n < _num; ++n) {
		p_data_ = s_data() + n * _v.size();
		cacu_copy_cpu(p_data_, _v.size(), &_v[0]);
		for (auto w : _v) os.write((char*)(&w), sizeof(w));
	}
	//vec_t().swap(_v);
#endif
}

/*
 * loads blob data from model file
 */
void weight::load_group(std::ifstream& is, int group) {
	float_t* p_data_ = (float_t*) _s_data;
#if __USE_DEVICE__ == ON
	vec_t _v(_channel_length * (_channel / group));
	int length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(int));
	CHECK_EQ_OP(length_, _num * _v.size(),
			"parameter '%s' length is not equal to local weight: %d vs %d!",
			_name.c_str(), length_, _num * _v.size());
	for (int n = 0; n < _num; ++n) {
		for (int i = 0; i < _v.size(); ++i)
			is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
		p_data_ = s_data() + n * _v.size();
		device_copy2dev(p_data_, &_v[0], _v.size());
	}
	//vec_t().swap(_v);
#else
	vec_t _v(_channel_length * (_channel / group));
	int length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(int));
	CHECK_EQ_OP(length_,_num * _v.size(),"parameter '%s' length is not equal to local weight: %d vs %d!", _name.c_str(), length_, _num * _v.size());
	for (int n = 0; n < _num; ++n) {
		for(int i = 0; i < _v.size(); ++i)
		is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
		p_data_ = s_data() + n * _v.size();
		cacu_copy_cpu(&_v[0], _v.size(), p_data_);
	}
	//vec_t().swap(_v);
#endif
}

void weight::load(std::ifstream& is) {
	dsize_t length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(dsize_t));
	CHECK_EQ_OP(length_, _length,
			"parameter [%s] length is not equal to local length: %d vs %d!",
			_name.c_str(), length_, _length);
	_tdata->load(is);
}

}
