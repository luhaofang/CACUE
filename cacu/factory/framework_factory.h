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

#include "../framework/blob_base.h"
#include "../framework/blob.h"
#include "../framework/bin_blob.h"

namespace cacu {

class cacu_allocator {

public:

	inline static blob* create_blob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, float_t value, phase_type phase) {
		return new blob(num, channel, width, height, value, phase);
	}

	inline static blob* create_blob(dsize_t num, dsize_t channel, dsize_t width,
			dsize_t height, phase_type phase) {
		return new blob(num, channel, width, height, 0, phase);
	}

	inline static blob* create_blob(blob_base *&blob_) {
		return new blob(blob_->num(), blob_->channel(), blob_->width(),
				blob_->height(), 0, blob_->phase());
	}

	inline static blob* create_blob(blob_base *&blob_, phase_type phase_) {
		return new blob(blob_->num(), blob_->channel(), blob_->width(),
				blob_->height(), 0, phase_);
	}

	inline static bin_blob* create_bin_blob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, unsigned int value, phase_type phase) {
		return new bin_blob(num, channel, width, height, value, phase);
	}

	inline static bin_blob* create_bin_blob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase) {
		return new bin_blob(num, channel, width, height, 0, phase);
	}

	inline static bin_blob* create_bin_blob(blob_base *&bin_blob_) {
		return new bin_blob(bin_blob_->num(), bin_blob_->channel(),
				bin_blob_->width(), bin_blob_->height(), 0, bin_blob_->phase());
	}

	inline static bin_blob* create_bin_blob(blob_base *&bin_blob_,
			phase_type phase_) {
		return new bin_blob(bin_blob_->num(), bin_blob_->channel(),
				bin_blob_->width(), bin_blob_->height(), 0, phase_);
	}

	/*
	inline static em_blob* create_em_blob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, float_t value, phase_type phase) {
		return new em_blob(num, channel, width, height, value, phase);
	}

	inline static em_blob* create_em_blob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase) {
		return new em_blob(num, channel, width, height, 0, phase);
	}

	inline static em_blob* create_em_blob(blob_base *&blob_) {
		return new em_blob(blob_->num(), blob_->channel(), blob_->width(),
				blob_->height(), 0, blob_->phase());
	}

	inline static em_blob* create_em_blob(blob_base *&blob_,
			phase_type phase_) {
		return new em_blob(blob_->num(), blob_->channel(), blob_->width(),
				blob_->height(), 0, phase_);
	}

	inline static em_bin_blob* create_em_bin_blob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, unsigned int value, phase_type phase) {
		return new em_bin_blob(num, channel, width, height, value, phase);
	}

	inline static em_bin_blob* create_em_bin_blob(dsize_t num, dsize_t channel,
			dsize_t width, dsize_t height, phase_type phase) {
		return new em_bin_blob(num, channel, width, height, 0, phase);
	}

	inline static em_bin_blob* create_em_bin_blob(blob_base *&bin_blob_) {
		return new em_bin_blob(bin_blob_->num(), bin_blob_->channel(),
				bin_blob_->width(), bin_blob_->height(), 0, bin_blob_->phase());
	}

	inline static em_bin_blob* create_em_bin_blob(blob_base *&bin_blob_,
			phase_type phase_) {
		return new em_bin_blob(bin_blob_->num(), bin_blob_->channel(),
				bin_blob_->width(), bin_blob_->height(), 0, phase_);
	}
	*/
	inline static blobs* create_blobs() {
		return new blobs();
	}


#define FREE_EDGE(edge_) {edge_->s_data = NULL;};

};

}
