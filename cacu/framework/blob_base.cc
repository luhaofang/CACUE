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


#include "blob_base.h"
#include "blob_definition.h"

namespace cacu {

blob_base::blob_base(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
		phase_type phase, blob_type type) {

	_width = width;
	_height = height;
	_channel = channel;
	_num = num;
	_channel_length = width * height;
	_cube_length = channel * width * height;
	_length = _num * _cube_length;
	_phase = phase;
	_blob_type = type;
	_s_data = NULL;
	_s_diff = NULL;
	_data_num = _num;
	_REC_ = 0;
	//check the data dim
	CHECK_LT_OP(_length, numeric_limits<dsize_t>::max(),
			"BLOB cell length is out of bounds: %ld vs d", _length,
			numeric_limits<dsize_t>::max());
}

blob_base::~blob_base() {
	_s_data = NULL;
	_s_diff = NULL;
}

void blob_base::_CHECK_SIZE_EQ(blob_base* blob_) {
	CHECK_EQ_OP(_channel, blob_->_channel,
			"_channel check does NOT match! ( %d vs %d )", _channel,
			blob_->_channel);
	CHECK_EQ_OP(_num, blob_->_num, "_num check does NOT match! ( %d vs %d )",
			_num, blob_->_num);
	CHECK_EQ_OP(_height, blob_->_height,
			"_height check does NOT match! ( %d vs %d )", _height,
			blob_->_height);
	CHECK_EQ_OP(_width, blob_->_width,
			"_width check does NOT match! ( %d vs %d )", _width, blob_->_width);
}

void blob_base::_REC() {
	if (_REC_ < __BLOBMAXREC__)
		_REC_ += 1;
	else
		LOG_WARNING(
				"blob address %X has rec for %d times, but the max rec for cacu is %d",
				this, _REC_, __BLOBMAXREC__);
}

}
