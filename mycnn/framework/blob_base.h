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

#pragma once

#include <vector>
#include <iosfwd>
#include <fstream>
#include <istream>
#include <ostream>

#include "blob_definition.h"
#include "../math/utils/math_utils.h"

using namespace std;

namespace mycnn {

class blob_base {

public:

	blob_base(size_t num, size_t channel, size_t width, size_t height,
			phase_type phase, blob_type type);

	virtual ~blob_base();

	inline size_t num() const {
		return _num;
	}

	inline size_t channel() const {
		return _channel;
	}

	inline size_t width() const {
		return _width;
	}

	inline size_t height() const {
		return _height;
	}

	inline size_t channel_length() const {
		return _channel_length;
	}

	inline size_t length() const {
		return _cube_length;
	}

	inline size_t count() const {
		return _length;
	}

	inline size_t data_num() const {
		return _data_num;
	}

	inline phase_type phase() const {
		return _phase;
	}

	inline void set_data_num(size_t num) {
		_data_num = num;
	}

	virtual inline size_t calculate_size() = 0;

	virtual inline const void _RESET_DATA() = 0;

	virtual inline const void _RESET_DIFF() = 0;

	virtual const void serializa(std::ostream& os) = 0;

	virtual const void load(std::ifstream& is) = 0;

	virtual inline const void resize(size_t num, size_t channel, size_t width,
			size_t height) = 0;

	inline size_t index(size_t c, size_t x, size_t y) const {
		return c * _cube_length + x * _width + y;
	}

	inline void _CHECK_SIZE_EQ(blob_base* blob_);

	inline blob_type _TYPE() const {
		return _blob_type;
	}

	inline bool is_output() const {
		return _REC_ == 0;
	}

	inline void _REC();

protected:

	size_t _width;
	size_t _height;
	size_t _channel;
	size_t _num;
	size_t _cube_length;
	size_t _length;
	size_t _channel_length;

	size_t _data_num;
	phase_type _phase;

	void *_s_data;

	void *_s_diff;

private:

	blob_type _blob_type;

	unsigned _REC_;

};
}
