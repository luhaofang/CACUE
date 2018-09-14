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

#include "../math/utils/rand_t.h"
#include "../utils/data_printer.h"


using namespace std;

namespace cacu {

blob::blob(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
		float_t _value, phase_type phase, bool _malloc_when_init) :
		blob_base(num, channel, width, height, phase, __blob__) {
	_init_value = _value;
	if(_malloc_when_init)
		_MALLOC();
}

blob::~blob() {
	delete _tdata;
	if (_tdiff != NULL) {
		delete _tdiff;
	}
}

/**
 * copy dest blob data to local blob
 */
void blob::copy_blob(blob* blob_) {
	this->_CHECK_SIZE_EQ(blob_);
	_tdata->copy2data(blob_->s_data());
	CHECK_EQ_OP(blob_->phase(), _phase, "blob phase must be the same! %d vs %d",
			blob_->phase(), _phase);
	if (train == _phase && train == blob_->phase()) {
		//_tdiff->copy2data(blob_->s_diff());
		_tdiff->set_value(0);
	}
}

blob* blob::copy_create(phase_type phase_, float_t value_) const {
	return new blob(num(), channel(), width(), height(), value_, phase_);
}

/*
 * copy data dsize_to blob, if blob is established in gpu, io op is needed
 * where i is the start index in blob
 */
void blob::copy2data(vec_t &data_, dsize_t i) {
	CHECK_EQ_OP(data_.size(), length(), "blob size must be equal! %d vs %d",
			data_.size(), length());
	_tdata->copy2data(i*length(), length(), &data_[0]);
}

void blob::copy2data(float_t *data_, dsize_t i) {
	_tdata->copy2data(i*length(), length(), data_);
}

/*
 * copy data dsize_to blob, if blob is established in gpu, io op is needed
 */
void blob::copy2data(vec_t &data_) {
	CHECK_EQ_OP(data_.size(), count(), "blob size must be equal! %d vs %d",
			data_.size(), count());
	_tdata->copy2data(&data_[0]);
}

/*
 * copy data dsize_to blob, if blob is established in gpu, io op is needed
 * where i is the start index in blob
 */
void blob::copy2diff(vec_t &data_, dsize_t i) {
	CHECK_EQ_OP(data_.size(), length(), "blob size must be equal! %d vs %d",
			data_.size(), length());
	_tdiff->copy2data(i*length(), length(), &data_[0]);
}

/*
 * copy data dsize_to blob's diff, if blob is established in gpu, io op is needed
 */
void blob::copy2diff(vec_t &data_) {
	CHECK_EQ_OP(data_.size(), count(), "blob size must be equal! %d vs %d",
			data_.size(), count());
	_tdiff->copy2data(&data_[0]);
}

void blob::output_bin(chars_t path_)
{
	std::ofstream os(path_, ios::binary);
	os.precision(numeric_limits<float_t>::digits10);
	if (!os)
		LOG_FATAL("file %s cannot be opened!", path_.c_str());
#if __USE_DEVICE__ == ON
	vec_t _v(count());
	device_copy2host(&_v[0], (float_t*)_s_data, count());
	for (int i = 0; i < count(); ++i) {
			os.write((char*) (&_v[i]), sizeof(cacu::float_t));
		}
#else
	for (int i = 0; i < count(); ++i) {
		os.write((char*) (s_data() + i), sizeof(cacu::float_t));
	}
#endif
	os.close();
}

void blob::input_bin(chars_t path_, int n)
{
	std::ifstream is(path_, ios::binary);
	is.precision(numeric_limits<float_t>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", path_.c_str());
#if __USE_DEVICE__ == ON
	vec_t _v(length());
	for (int i = 0; i < length(); i++) {
		is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
	}
	device_copy2dev(p_data(n), &_v[0], length());
#else
	for (int i = 0; i < length(); i++) {
		is.read(reinterpret_cast<char*>(p_data(n) + i), sizeof(float_t));
	}
#endif
	is.close();
}

void blob::load_from(chars_t path_)
{
	std::ifstream is(path_, ios::binary);
	is.precision(numeric_limits<float_t>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", path_.c_str());
	string line = "";
#if __USE_DEVICE__ == ON
	vec_t _v(count());
	
	int i= 0;
	while(getline(is, line))
	{
		_v[i] = strtof(line.c_str(), NULL);
		i+=1;
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

/*
 * serializa blob data, output data to model file
 */
void blob::serializa(std::ostream& os) {
	_tdata->serializa(os);
}

/*
 * loads blob data from model file
 */
void blob::load(std::ifstream& is) {
	dsize_t length_;
	is.read(reinterpret_cast<char*>(&length_), sizeof(dsize_t));
	CHECK_EQ_OP(length_,count(),"parameter length is not equal to local length: %d vs %d!",length_,count());
	_tdata->load(is);
}

void blob::set_init_type(param_init_type type, float_t value) {
	vec_t w(count());
	switch (type) {
	case constant:
		for (int i = 0; i < count(); ++i)
			w[i] = value;
		break;
	case xavier:
		value = sqrt((float_t) 3.0 / (channel() * height() * width()));
		for (int i = 0; i < count(); ++i)
			w[i] = urand(-value, value);
		break;
	case gaussian:
		for (int i = 0; i < count(); ++i)
			w[i] = gaussrand(value);
		break;
	case msra:
		value = sqrt((float_t) 2.0 / (channel() * height() * width()));
		for (int i = 0; i < count(); ++i)
			w[i] = gaussrand(value);
		break;
	case uniform:
		for (int i = 0; i < count(); ++i)
			w[i] = urand(-value, value);
		break;
	default:
		break;
	}
	_tdata->copy2data(&w[0]);
	vec_t().swap(w);
}

void blob::switch_channel()
{
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	vec_t temp(count());
	cuda_copy2host(&temp[0],s_data(),count());
	cacu_transpose(&temp[0],num(), channel(), channel_length());
	cuda_copy2dev(s_data(),&temp[0],count());
	if(_phase == train){
		cuda_copy2host(&temp[0],s_diff(),count());
		cacu_transpose(&temp[0], num(), channel(), channel_length());
		cuda_copy2dev(s_diff(),&temp[0],count());
	}
#endif
#else
	cacu_transpose(s_data(), num(), channel(), channel_length());
	if(_phase == train)
		cacu_transpose(s_diff(), num(), channel(), channel_length());
#endif

	int temp_ = num();
	_body->_num = channel();
	_body->_channel = temp_;
	_body->_cube_length = _body->_channel * _body->_channel_length;
}

}
