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

#pragma once

#include "../definition.h"

#include "../utils/log.h"
#include "../utils/check_utils.h"

#include "blob_definition.h"

using namespace std;

namespace cacu {

class blob_base {

public:

	blob_base(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
			phase_type phase, blob_type type);

	virtual ~blob_base();

	inline dsize_t num() const {
		return _num;
	}

	inline dsize_t channel() const {
		return _channel;
	}

	inline dsize_t width() const {
		return _width;
	}

	inline dsize_t height() const {
		return _height;
	}

	inline dsize_t channel_length() const {
		return _channel_length;
	}

	inline dsize_t length() const {
		return _cube_length;
	}

	inline dsize_t count() const {
		return _length;
	}

	inline dsize_t data_num() const {
		return _data_num;
	}

	inline phase_type phase() const {
		return _phase;
	}

	inline void set_data_num(dsize_t num) {
		_data_num = num;
	}

	virtual void resize(dsize_t num, dsize_t channel, dsize_t width, dsize_t height) = 0;

	virtual dsize_t calculate_size() = 0;

	virtual void _RESET_DATA() = 0;

	virtual void _RESET_DIFF() = 0;

	virtual void serializa(std::ostream& os) = 0;

	virtual void load(std::ifstream& is) = 0;

	inline dsize_t index(dsize_t c, dsize_t x, dsize_t y) const {
		return c * _cube_length + x * _width + y;
	}

	void _CHECK_SIZE_EQ(blob_base* blob_);

	inline blob_type _TYPE() const {
		return _blob_type;
	}

	inline bool is_output() const {
		return _REC_ == 0;
	}

	void _REC();

	inline bool _IS_MOTIFIED() {
		if(_p_length == _length)
			return true;
		else
			_p_length = _length;
		return false;
	}

protected:

	dsize_t _width;
	dsize_t _height;
	dsize_t _channel;
	dsize_t _num;
	dsize_t _cube_length;
	dsize_t _length;
	dsize_t _channel_length;

	dsize_t _data_num;
	phase_type _phase;

	void *_s_data;

	void *_s_diff;

private:

	blob_type _blob_type;

	unsigned _REC_;

	dsize_t _p_length;

};
}
