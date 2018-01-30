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


#include "blob_base.h"

#include "../math/math.h"
#include "../definition.h"

#include "../tensor/tensor.h"

using namespace std;

namespace cacu{

	class blob : public blob_base{

	public:

		blob(size_t num, size_t channel, size_t width, size_t height, float_t _value, phase_type phase);

		~blob();



		/**
		 * return the piece probe in blob data
		 */
		inline float_t* p_data(size_t n) const{
			CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num);
			return (float_t*)_s_data + n*_cube_length;
		}

		/**
		 * return the piece probe in blob diff
		 */
		inline float_t* p_diff(size_t n) const{
			CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num);
			return (float_t*)_s_diff + n*_cube_length;
		}

		/**
		 * return the source probe in blob data
		 */
		inline float_t* s_data() const{ return (float_t*)_s_data; }

		/**
		 * return the source probe in blob diff
		 */
		inline float_t* s_diff() const{ return (float_t*)_s_diff; }
		/**
		 * copy dest blob data to local blob
		 */
		inline void copy_blob(blob* blob_);

		inline blob* copy_create(phase_type phase_, float_t value_) const;

		/*
		 * copy data size_to blob, if blob is established in gpu, io op is needed
		 * where i is the start piece index in blob
		 */
		inline void copy2data(vec_t &data_, size_t i);

		/*
		 * copy data size_to blob, if blob is established in gpu, io op is needed
		 */
		inline void copy2data(vec_t &data_);

		/*
		 * copy data size_to blob, if blob is established in gpu, io op is needed
		 * where i is the start piece index in blob
		 */
		inline void copy2diff(vec_t &data_, size_t i);

		/*
		 * copy data size_to blob's diff, if blob is established in gpu, io op is needed
		 */
		inline void copy2diff(vec_t &data_);

		virtual inline size_t calculate_size() = 0;

		virtual inline const void _RESET_DATA() = 0;

		virtual inline const void _RESET_DIFF() = 0;

		virtual const void serializa(std::ostream& os) = 0;

		virtual const void load(std::ifstream& is) = 0;

		virtual inline const void resize(size_t num, size_t channel, size_t width, size_t height) = 0;


	private:

		tensor<float_t>* _tdata;

		tensor<float_t>* _tdiff;


	};
}
