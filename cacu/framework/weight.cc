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
#include "../math/math.h"
#include "../utils/data_printer.h"

namespace cacu {

weight::weight(const chars_t& name, dsize_t num, dsize_t channel, dsize_t width,
		dsize_t height, phase_type phase) :
		blob(num, channel, width, height, 0, phase, true) {
	_name = name;
	_update_lr = 1.0;
	_decay_mult = 1.0;
	_update = true;
	_variable = true;
	_upgrade_index = new vec_i();
	_update_index = new vec_i();
#if __USE_CUDNN__ == ON
	create_filter_desc(_filter_desc);
	set_filter_4d_desc(_filter_desc, _body->_num, _body->_channel, _body->_width, _body->_height);
#endif
}

weight::weight(chars_t&& name, dsize_t num, dsize_t channel, dsize_t width,
		dsize_t height, phase_type phase) :
		blob(num, channel, width, height, 0, phase, true) {
	_name = std::move(name);
	_update_lr = 1.0;
	_decay_mult = 1.0;
	_update = true;
	_variable = true;
	_upgrade_index = new vec_i();
	_update_index = new vec_i();
#if __USE_CUDNN__ == ON
	create_filter_desc(_filter_desc);
	set_filter_4d_desc(_filter_desc, _body->_num, _body->_channel, _body->_width, _body->_height);
#endif
}

weight::~weight() {
#if __USE_CUDNN__ == ON
	destroy_filter_descriptor(_filter_desc);
#endif
	delete _upgrade_index;
	delete _update_index;
}


/*
 * serializa blob data, output data to model file
 */
void weight::serializa_group(std::ostream& os, int group) {
	float_t* p_data_;
	int length = count() / group;
#if __USE_DEVICE__ == ON
	os.write((char*) (&length), sizeof(length));
	vec_t _v(channel_length() * (channel() / group));
	for (int n = 0; n < num(); ++n) {
		p_data_ = s_data() + n * _v.size();
		device_copy2host(&_v[0], p_data_, _v.size());
		for (auto w : _v)
			os.write((char*) (&w), sizeof(w));
	}
	vec_t().swap(_v);
#else

	os.write((char*)(&length), sizeof(length));
	vec_t _v(channel_length() * (channel() / group));
	for(int n = 0; n < num(); ++n) {
		p_data_ = s_data() + n * _v.size();
		cacu_copy_cpu(p_data_, _v.size(), &_v[0]);
		for (auto w : _v) os.write((char*)(&w), sizeof(w));
	}
	vec_t().swap(_v);
#endif
}

/*
 * loads blob data from model file
 */
void weight::load_group(std::ifstream& is, int group) {
	float_t* p_data_ = (float_t*) _s_data;
#if __USE_DEVICE__ == ON
	vec_t _v(channel_length() * (channel() / group));
	int length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(int));
	CHECK_EQ_OP(length_, num() * _v.size(),
			"parameter '%s' length is not equal to local weight: %d vs %d!",
			_name.c_str(), length_, num() * (int)_v.size());
	for (int n = 0; n < num(); ++n) {
		for (int i = 0; i < _v.size(); ++i)
			is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
		p_data_ = s_data() + n * _v.size();
		device_copy2dev(p_data_, &_v[0], _v.size());
	}
	vec_t().swap(_v);
#else
	vec_t _v(channel_length() * (channel() / group));
	int length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(int));
	CHECK_EQ_OP(length_,num() * (int)_v.size(),"parameter '%s' length is not equal to local weight: %d vs %d!",
			_name.c_str(), length_, num() * (int)_v.size());
	for (int n = 0; n < num(); ++n) {
		for(int i = 0; i < _v.size(); ++i)
		is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
		p_data_ = s_data() + n * _v.size();
		cacu_copy_cpu(&_v[0], _v.size(), p_data_);
	}
	vec_t().swap(_v);
#endif
}

void weight::load(std::ifstream& is) {
	dsize_t length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(dsize_t));
	CHECK_EQ_OP(length_, count(),
			"parameter [%s] length is not equal to local length: %d vs %d!",
			_name.c_str(), length_, count());
	_tdata->load(is);
}


}
