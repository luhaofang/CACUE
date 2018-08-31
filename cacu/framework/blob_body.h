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

#ifndef BLOB_BODY_H_
#define BLOB_BODY_H_

namespace cacu{

class blob_body
{
	public:

	blob_body(dsize_t num, dsize_t channel, dsize_t width, dsize_t height){
		_num = num;
		_channel = channel;
		_width = width;
		_height = height;
		_channel_length = width * height;
		_cube_length = channel * width * height;
		_length = _num * _cube_length;
	}

	~blob_body(){}

	dsize_t _width;
	dsize_t _height;
	dsize_t _channel;
	dsize_t _num;
	dsize_t _cube_length;
	dsize_t _length;
	dsize_t _channel_length;

	inline bool check_body(blob_body *body_)
	{
		if(body_->_num != _num)
			return false;
		if(body_->_channel != _channel)
			return false;
		if(body_->_width != _width)
			return false;
		if(body_->_height != _height)
			return false;
		return true;
	}

	inline void copy_from(blob_body *body_)
	{
		_num = body_->_num;
		_channel = body_->_channel;
		_width = body_->_width;
		_height = body_->_height;
		_channel_length = _width * _height;
		_cube_length = _channel * _width * _height;
		_length = _num * _cube_length;
	}

	inline void set_body(dsize_t num_, dsize_t channel_, dsize_t width_, dsize_t height_)
	{
		_num = num_;
		_channel = channel_;
		_width = width_;
		_height = height_;
		_channel_length = _width * _height;
		_cube_length = _channel * _width * _height;
		_length = _num * _cube_length;
	}

};

}


#endif /* CACU_FRAMEWORK_BLOB_BODY_H_ */
