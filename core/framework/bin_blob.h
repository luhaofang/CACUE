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

#include <bitset>

#include "../math/cuda/cuda_utils.h"

using namespace std;

namespace mycnn{

	class bin_blob : public blob_base{

	public:

		bin_blob(int num, int channel, int width, int height, unsigned int _value, phrase_type phrase)
			:blob_base(num, channel, width, height, phrase, __bin_blob__){
#if __PARALLELTYPE__ == __CUDA__
			_s_data = cuda_malloc_v<unsigned int>(_num,_cube_length,_value);
			if (train == phrase){
				_s_diff = cuda_malloc_v<float_t>(_num,_cube_length, 0);
			}
#else
			_s_data = (unsigned int*)malloc(_length * sizeof(unsigned int));
			set_data(_value);
			if (train == phrase){
				_s_diff = (float_t*)malloc(_length * sizeof(float_t));
				set_diff(0);
			}
#endif

		}

		~bin_blob(){

		}

		inline unsigned int* p_data(int n) const{
			CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num);
			return (unsigned int*)_s_data + n*_cube_length;
		}

		inline float_t* p_diff(int n) const{
			CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num);
			return (float_t*)_s_diff + n*_cube_length;
		}

		inline unsigned int* s_data() const{ return (unsigned int*)_s_data; }

		inline float_t* s_diff() const{ return (float_t*)_s_diff; }

		virtual inline const void _RESET_DATA() override
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;
			float_t* s_diff_ = (float_t*)_s_diff;
#if __PARALLELTYPE__ == __CUDA__
			cuda_refresh(s_data_, _length);
			if (train == _phrase)
				cuda_refresh(s_diff_,_length);
#else
			for(int i = 0 ; i < _length ; ++i)
				s_data_[i] = 0;
			if (train == _phrase)
				for(int i = 0 ; i < _length ; ++i)
					s_diff_[i] = 0.0;
#endif
		}

		virtual inline const void _RESET_DIFF() override
		{
			float_t* s_diff_ = (float_t*)_s_diff;
#if __PARALLELTYPE__ == __CUDA__
			if (train == _phrase)
				cuda_refresh(s_diff_,_length);
#else
			if (train == _phrase)
				for(int i = 0 ; i < _length ; ++i)
					s_diff_[i] = 0.0;
#endif
		}

		inline void set_data(unsigned int value_)
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;
#if __PARALLELTYPE__ == __CUDA__
			cuda_setvalue<unsigned int>(s_data_, value_,_length);
#else
			for(int i = 0 ; i < _length ; ++i)
				s_data_[i] = value_;
#endif
		}

		inline void set_diff(float_t value_)
		{
			float_t* s_diff_ = (float_t*)_s_diff;
			if (train == _phrase){
#if __PARALLELTYPE__ == __CUDA__
				cuda_setvalue<float_t>(s_diff_, value_,_length);
#else
				for(int i = 0 ; i < _length ; ++i)
					s_diff_[i] = value_;
#endif
			}
		}

		inline virtual const int calculate_size() override{
			return test == _phrase ? _length*sizeof(unsigned int) : _length*sizeof(unsigned int) + _length*sizeof(float_t);
		}

		inline void copy_data_io(vec_i &data_ , int i)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
#if __PARALLELTYPE__ == __CUDA__
			cuda_copy2dev(p_data(i),&data_[0],_cube_length);
#else
			memcpy(p_data(i),&data_[0],_cube_length*sizeof(unsigned int));
#endif
		}

		inline void copy_data_io(vec_i &data_)
		{
			CHECK_EQ_OP(data_.size(),_length,"blob size must be equal! %d vs %d",data_.size(),_length);
#if __PARALLELTYPE__ == __CUDA__
			cuda_copy2dev(s_data(),&data_[0],_length);
#else
			memcpy(s_data(),&data_[0],_length*sizeof(unsigned int));
#endif
		}

		inline void copy_data_io(vec_t &data_, int i)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
#if __PARALLELTYPE__ == __CUDA__
			cuda_copy2dev(p_diff(i),&data_[0],_cube_length);
#else
			memcpy(p_diff(i),&data_[0],_cube_length*sizeof(float_t));
#endif
		}

		/*
		 * copy data into blob's diff, if blob is established in gpu, io op is needed
		 */
		inline void copy_diff_io(vec_t &data_)
		{
			CHECK_EQ_OP(data_.size(),_length,"blob size must be equal! %d vs %d",data_.size(),_length);
#if __PARALLELTYPE__ == __CUDA__
			cuda_copy2dev(s_diff(), &data_[0], _length);
#else
			memcpy(s_diff(), &data_[0],_length*sizeof(float_t));
#endif
		}

		/*
		 * serializa blob_bit data, output data to model file
		 * warning: bin_blob just serializa bit wise parameters.
		 */
		inline virtual const void serializa(std::ostream& os) override
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;
			bitset<32> _bits;
#if __PARALLELTYPE__ == __CUDA__
			int _len = _length / 32;
			if(_length % 32 != 0)
				_len += 1;
			os.write((char*)(&_len), sizeof(_len));
			vec_t _v(_length);
			cuda_copy2host(&_v[0],(float_t*)_s_data,_length);
			int _index;

			for(int i = 0 ; i < _v.size(); ++i)
			{
				_index = i % 32;
				if (_v[i])
					_bits[_index] = 1;
				else
					_bits[_index] = 0;
				if(_index == 31 || i == (_v.size() - 1)){
					auto w = _bits.to_ulong();
					os.write((char*)(&w), sizeof(w));
				}
			}
			vec_t().swap(_v);
#else
			int _len = _length / 32;
			if(_length % 32 != 0)
				_len += 1;
			os.write((char*)(&_len), sizeof(_len));
			int _index;
			for(int i = 0 ; i < _length; ++i)
			{
				_index = i % 32;
				if (s_data_[i])
					_bits[_index] = 1;
				else
					_bits[_index] = 0;
				if(_index == 31 || i == (_length - 1)){
					auto w = _bits.to_ulong();
					os.write((char*)(&w), sizeof(w));
				}
			}
#endif
		}

		/*
		 * load blob_bit data from model file
		 */
		inline virtual const void load(std::ifstream& is) override
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;
#if __PARALLELTYPE__ == __CUDA__
			int length_;
			is.read(reinterpret_cast<char*>(&length_), sizeof(int));
			int _len = _length / 32;
			if(_length % 32 != 0)
				_len += 1;
			CHECK_EQ_OP(length_,_length,"parameter length is not equal to local weight: %d vs %d!",length_,_len);
			vec_t _v(_length);
			unsigned int _bit;
			for (int i = 0; i < _len; i++){
				is.read(reinterpret_cast<char*>(&_bit), sizeof(unsigned int));
				bitset<32> bits(_bit);
				for(int j = 0; j < 32; ++j)
				{
					if(bits.test(j) && (i * 32 + j < _length))
						_v[i * 32 + j] = 1;
					else if(!bits.test(j) && (i * 32 + j < _length))
						_v[i * 32 + j] = -1;
				}
			}
			cuda_copy2dev((float_t*)_s_data, &_v[0],length_);
			vec_t().swap(_v);
#else
			int length_;
			is.read(reinterpret_cast<char*>(&length_), sizeof(int));
			int _len = _length / 32;
			if(_length % 32 != 0)
				_len += 1;
			CHECK_EQ_OP(length_,_length,"parameter length is not equal to local weight: %d vs %d!",length_,_len);
			unsigned int _bit;
			for (int i = 0; i < _len; i++){
				is.read(reinterpret_cast<char*>(&_bit), sizeof(unsigned int));
				bitset<32> bits(_bit);
				for(int j = 0; j < 32; ++j)
				{
					if(bits.test(j) && (i * 32 + j < _length))
						s_data_[i * 32 + j] = 1;
					else if(!bits.test(j) && (i * 32 + j < _length))
						s_data_[i * 32 + j] = -1;
				}
			}
#endif
		}

		inline virtual const void resize(int num, int channel, int width, int height) override
		{
			_width = width;
			_height = height;
			_channel = channel;
			_num = num;
			_channel_length = width*height;
			_cube_length = channel*width*height;
			_length = _num * _cube_length;

#if __PARALLELTYPE__ == __CUDA__
			cuda_free(_s_data);
			_s_data = cuda_malloc_v<unsigned int>(_num,_cube_length, 0);
			if (train == _phrase){
				cuda_free(_s_diff);
				_s_diff = cuda_malloc_v<float_t>(_num,_cube_length, 0);
			}
#else
			free(_s_data);
			_s_data = (unsigned int*)malloc(_length * sizeof(unsigned int));
			set_data(0);
			if (train == _phrase){
				free(_s_diff);
				_s_diff = (float_t*)malloc(_length * sizeof(float_t));
				set_diff(0);
			}
#endif
		}


	protected:

	};
}
