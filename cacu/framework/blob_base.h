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

#ifndef BLOB_BASE_H_
#define BLOB_BASE_H_

#include "../definition.h"

#include "../utils/log.h"
#include "../utils/check_utils.h"

#include "blob_definition.h"
#include "blob_body.h"
#include "../math/cudnn/cudnn_functions.h"

using namespace std;

namespace cacu {

class blob_base {

public:

	blob_base(dsize_t num, dsize_t channel, dsize_t width, dsize_t height,
			phase_type phase, blob_type type);

	virtual ~blob_base();

	inline dsize_t num() const {
		return _body->_num;
	}

	inline dsize_t channel() const {
		return _body->_channel;
	}

	inline dsize_t width() const {
		return _body->_width;
	}

	inline dsize_t height() const {
		return _body->_height;
	}

	inline dsize_t channel_length() const {
		return _body->_channel_length;
	}

	inline dsize_t length() const {
		return _body->_cube_length;
	}

	inline dsize_t count() const {
		return _body->_length;
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

	inline blob_body *body() {
		return _body;
	}

	inline void copy_body(const blob_body *body_) {
		_body->copy_from(body_);
	}

	virtual void resize(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height) = 0;

	virtual dsize_t calculate_size() = 0;

	virtual void _RESET_DATA() = 0;

	virtual void _RESET_DIFF() = 0;

	virtual void serializa(std::ostream& os) = 0;

	virtual void load(std::ifstream& is) = 0;

	inline dsize_t index(dsize_t n, dsize_t c, dsize_t h, dsize_t w) const {
		return n * _body->_cube_length + c * _body->_channel_length
				+ h * _body->_width + w;
	}

	inline dsize_t index0(dsize_t h, dsize_t w) const {
		return h * _body->_width + w;
	}

	void _CHECK_SIZE_EQ(const blob_base* blob_);

	inline blob_type _TYPE() const {
		return _blob_type;
	}

	inline bool is_output() const {
		return _REC_ == 0;
	}

	/*
	 *
	 * whether does the blob can be RESET in _RESET_DATA && _RESET_DIFF
	 */
	inline void set_variable(bool type_) {
		_variable = type_;
	}

	void _REC();

	/*
	 * compare the current data size with previous data size
	 */
	inline bool _IS_MOTIFIED() {
		return _body->check_body(_p_body);
	}

	inline void _MOTIFY() {
		_p_body->copy_from(_body);
	}

	inline void _NEED_MOTIFY() {
		_p_body->set_body(0, 0, 0, 0);
	}

	void blob_size() {
		switch (_blob_type) {

		case __blob__:
			LOG_INFO("blob size: (%d, %d, %d, %d)", _body->_num,
					_body->_channel, _body->_width, _body->_height);
			break;
		case __bin_blob__:
			LOG_INFO("bin blob size: (%d, %d, %d, %d)", _body->_num,
					_body->_channel, _body->_width, _body->_height);
			break;
		default:
			break;
		}
	}

protected:

	blob_body *_body;

	dsize_t _data_num;
	phase_type _phase;

	bool _variable = true;

	int _REC_;

	void *_s_data;

	void *_s_diff;

private:

	blob_type _blob_type;

	blob_body *_p_body;

};
}

#endif
