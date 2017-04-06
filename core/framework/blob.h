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


#include "../math/cuda/cuda_utils.h"
#include "../math/math_utils.h"
#include "../math/math_functions.h"


using namespace std;

namespace mycnn{

	class blob : public blob_base{

	public:

		blob(int num, int channel, int width, int height, float_t _value, phrase_type phrase)
			:blob_base(num, channel, width, height, phrase, __blob__){

#if __PARALLELTYPE__ == __GPU__
			_s_data = cuda_malloc_v<float_t>(_num,_cube_length,_value);
			CUDA_CHECK(res);
			if (train == phrase){
				_s_diff = cuda_malloc_v<float_t>(_num,_cube_length,0);
				CUDA_CHECK(res);
			}
#else
			_s_data = (float_t*)malloc(_length * sizeof(float_t));
			set_data(_value);
			if (train == phrase){
				_s_diff = (float_t*)malloc(_length * sizeof(float_t));
				set_diff(0);
			}
#endif
		}

		~blob(){

		}

		/**
		 * return the piece probe in blob data
		 */
		inline float_t* p_data(int n) {
			return (float_t*)_s_data + n*_cube_length;
		}

		/**
		 * return the piece probe in blob diff
		 */
		inline float_t* p_diff(int n) {
			return (float_t*)_s_diff + n*_cube_length;
		}

		/**
		 * return the source probe in blob data
		 */
		inline float_t* s_data(){ return (float_t*)_s_data; }

		/**
		 * return the source probe in blob diff
		 */
		inline float_t* s_diff(){ return (float_t*)_s_diff; }

		/**
		 * reset all data (data&diff) in this blob
		 */
		inline virtual const void _RESET_DATA() override
		{
			float_t* s_data_ = (float_t*)_s_data;
			float_t* s_diff_ = (float_t*)_s_diff;
#if __PARALLELTYPE__ == __GPU__
			cuda_refresh(s_data_,_length);
			if (train == _phrase)
				cuda_refresh(s_diff_,_length);
#else
			memset(s_data_, 0 ,_length*sizeof(float_t));
			if (train == _phrase)
				memset(s_diff_, 0 ,_length*sizeof(float_t));
#endif
		}

		/**
		 * reset diff data (diff) in this blob
		 */
		inline virtual const void _RESET_DIFF() override
		{
			float_t* s_diff_ = (float_t*)_s_diff;
#if __PARALLELTYPE__ == __GPU__
			if (train == _phrase)
				cuda_refresh(s_diff_, _length);
#else
			if (train == _phrase)
				memset(s_diff_, 0 ,_length*sizeof(float_t));
#endif
		}

		/**
		 * set input diff data to constant value
		 */
		inline const void set_data(float_t value_)
		{
			float_t* s_data_ = (float_t*)_s_data;
#if __PARALLELTYPE__ == __GPU__
			cuda_setvalue<float_t>(s_data_,value_,_length);
#else
			for(int i = 0 ; i < _length ; ++i)
				s_data_[i] = value_;
#endif
		}

		/**
		 * set input diff data to constant value
		 */
		inline const void set_diff(float_t value_)
		{
			float_t* s_diff_ = (float_t*)_s_diff;
			if (train == _phrase){
#if __PARALLELTYPE__ == __GPU__
				cuda_setvalue<float_t>(s_diff_,value_,_length);
#else
				for(int i = 0 ; i < _length ; ++i)
					s_diff_[i] = value_;
#endif
			}
		}

		virtual inline const int calculate_size() override{
			return test == _phrase ? _length * sizeof(float_t) : 2 * _length * sizeof(float_t);
		}
		
		/**
		 * copy dest blob data to local blob
		 */
		inline const void copy_blob(blob* blob_)
		{
			CHECK_EQ_OP(blob_->count(),_length,"blob size must be equal! %d vs %d",blob_->count(),_length);
			cacu_copy(blob_->s_data(),_length, (float_t*)_s_data);
			if(train == _phrase && train == blob_->phrase()){
				cacu_copy(blob_->s_diff(),_length, (float_t*)_s_diff);
			}
		}

		inline blob* copy_create(phrase_type phrase_, float_t value_ = 0)
		{
			return new blob(_num, _channel, _width, _height, value_, phrase_);
		}

		/*
		 * copy data into blob, if blob is established in gpu, io op is needed
		 * where i is the start index in blob
		 */
		inline const void copy_data_io(vec_t &data_, int i)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
#if __PARALLELTYPE__ == __GPU__
			cuda_copy2dev(p_data(i), &data_[0], _cube_length);
#else
			cacu_copy(&data_[0],_cube_length, p_data(i));
#endif
		}

		/*
		 * copy data into blob, if blob is established in gpu, io op is needed
		 */
		inline const void copy_data_io(vec_t &data_)
		{
			CHECK_EQ_OP(data_.size(),_length,"blob size must be equal! %d vs %d",data_.size(),_length);
#if __PARALLELTYPE__ == __GPU__
			cuda_copy2dev(s_data(), &data_[0], _length);
#else
			cacu_copy(&data_[0],_length, s_data());
#endif
		}

		/*
		 * copy data into blob, if blob is established in gpu, io op is needed
		 * where i is the start index in blob
		 */
		inline const void copy_diff_io(vec_t &data_, int i)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
#if __PARALLELTYPE__ == __GPU__
			cuda_copy2dev(p_diff(i), &data_[0], _cube_length);
#else
			cacu_copy(&data_[0],_cube_length, p_diff(i));
#endif
		}

		/*
		 * copy data into blob's diff, if blob is established in gpu, io op is needed
		 */
		inline const void copy_diff_io(vec_t &data_)
		{
			CHECK_EQ_OP(data_.size(),_length,"blob size must be equal! %d vs %d",data_.size(),_length);
#if __PARALLELTYPE__ == __GPU__
			cuda_copy2dev(s_diff(), &data_[0], _length);
#else
			cacu_copy(&data_[0],_length, s_diff());
#endif
		}

		/*
		 * serializa blob data, output data to model file
		 */
		void serializa(std::ostream& os)
		{
			float_t* s_data_ = (float_t*)_s_data;
#if __PARALLELTYPE__ == __GPU__
			os.write((char*)(&_length), sizeof(_length));
			vec_t _v(_length);
			cuda_copy2host(&_v[0],(float_t*)_s_data,_length);
			for (auto w : _v) os.write((char*)(&w), sizeof(w));
			vec_t().swap(_v);
#else
			os.write((char*)(&_length), sizeof(_length));
			for(int i = 0 ; i < _length; ++i)
				os.write((char*)(&s_data_[i]), sizeof(s_data_[i]));
#endif
		}

		/*
		 * loads blob data from model file
		 */
		void load(std::ifstream& is)
		{
			float_t* s_data_ = (float_t*)_s_data;
#if __PARALLELTYPE__ == __GPU__
			vec_t _v(_length);
			int length_;
			is.read(reinterpret_cast<char*>(&length_), sizeof(int));
			CHECK_EQ_OP(length_,_length,"parameter length is not equal to local weight: %d vs %d!",length_,_length);
			for (int i = 0; i < length_; i++){
				is.read(reinterpret_cast<char*>(&_v[i]), sizeof(float_t));
			}
			cuda_copy2dev((float_t*)_s_data, &_v[0],length_);
			vec_t().swap(_v);
#else
			int length_;
			is.read(reinterpret_cast<char*>(&length_), sizeof(int));
			CHECK_EQ_OP(length_,_length,"parameter length is not equal to local weight: %d vs %d!",length_,_length);
			for (int i = 0; i < length_; i++){
				is.read(reinterpret_cast<char*>(s_data_ + i), sizeof(float_t));
			}
#endif
		}


	protected:

	};
}
