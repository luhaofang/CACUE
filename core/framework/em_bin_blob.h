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


#include "../math/device_data_utils.h"
#include "../math/cpu_data_utils.h"
#include "../math/math_utils.h"
#include "../math/math_functions.h"


using namespace std;

namespace mycnn{

	class em_bin_blob : public blob_base{

	public:

		em_bin_blob(int num, int channel, int width, int height, unsigned int _value, phrase_type phrase)
			:blob_base(num, channel, width, height, phrase, __em_bin_blob__){

			_data_length = min(__EMBEDSIZE__,num);

			_s_data = (float_t*)malloc(_length * sizeof(unsigned int));
			set_data(_value);
			_d_p_data = device_malloc_v<unsigned int>(_data_length,_cube_length,_value);

			if (train == phrase){
				_s_diff = (float_t*)malloc(_length * sizeof(float_t));
				set_diff(0);
				_d_p_diff = device_malloc_v<float_t>(_data_length,_cube_length,0);
			}

			__BUFFER__ = (int *) malloc(_data_length * sizeof(int));
		}

		~em_bin_blob(){
			free(__BUFFER__);

			device_free(_d_p_data);
			if(train == _phrase)
			{
				device_free(_d_p_diff);
			}
		}

		/**
		 * return the piece probe in blob data
		 */
		inline unsigned int* p_data(int n) {
			CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num);
			return (unsigned int*)_s_data + n *_cube_length;
		}

		/**
		 * return the piece probe in blob diff
		 */
		inline float_t* p_diff(int n) {
			CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num);
			return (float_t*)_s_diff + n *_cube_length;
		}

		/**
		 * return the source probe in blob data
		 */
		inline unsigned int* s_data(){ return (unsigned int*)_s_data; }

		/**
		 * return the source probe in blob diff
		 */
		inline float_t* s_diff(){ return (float_t*)_s_diff; }


		inline unsigned int* s_data_d()
		{
			return _d_p_data;
		}

		inline float_t* s_diff_d()
		{
			return _d_p_diff;
		}

		inline unsigned int* p_data_d(int i)
		{
			if(_num != 1){
				int buffer_index = __HITS__(i);
				//no hit in buffer, switch!
				if(FAILEDHITS == buffer_index)
				{
					buffer_index = switch_host2dev(i);
					__BUFFER__[buffer_index] = i;
				}
				return _d_p_data + buffer_index * _cube_length;
			}
			else{
				int buffer_index = __HITS__(i);
				if(FAILEDHITS == buffer_index)
				{
					buffer_index = switch_host2dev(i);
					__BUFFER__[buffer_index] = i;
				}
				return _d_p_data;
			}
		}

		inline float_t* p_diff_d(int i)
		{
			if(_num != 1){
				int buffer_index = __HITS__(i);
				//no hit in buffer, switch!
				if(FAILEDHITS == buffer_index)
				{
					buffer_index = switch_host2dev(i);
					__BUFFER__[buffer_index] = i;
				}
				return _d_p_diff + buffer_index * _cube_length;
			}
			else{
				int buffer_index = __HITS__(i);
				if(FAILEDHITS == buffer_index)
				{
					buffer_index = switch_host2dev(i);
					__BUFFER__[buffer_index] = i;
				}
				return _d_p_diff;
			}
		}

		/**
		 * reset all data (data & diff) in this blob
		 */
		inline virtual const void _RESET_DATA() override
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;
			float_t* s_diff_ = (float_t*)_s_diff;

			cacu_memset<unsigned int>(s_data_, 0 ,_length);
			if (train == _phrase)
				cacu_memset<mycnn::float_t>(s_diff_, 0 ,_length);
			cacu_memset<int>(__BUFFER__, FAILEDHITS, _data_length);
		}

		/**
		 * reset diff data (diff) in this blob
		 */
		inline virtual const void _RESET_DIFF() override
		{
			float_t* s_diff_ = (float_t*)_s_diff;

			if (train == _phrase)
				cacu_memset<mycnn::float_t>(s_diff_, 0 ,_length);
			cacu_memset<int>(__BUFFER__, FAILEDHITS, _data_length);
		}

		/**
		 * set input diff data to constant value
		 */
		inline const void set_data(unsigned int value_)
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;

			for(int i = 0 ; i < _length ; ++i)
				s_data_[i] = value_;
		}

		/**
		 * set input diff data to constant value
		 */
		inline const void set_diff(float_t value_)
		{
			float_t* s_diff_ = (float_t*)_s_diff;

			if (train == _phrase){
				for(int i = 0 ; i < _length ; ++i)
					s_diff_[i] = value_;
			}
		}

		/*
		 * copy data into blob, if blob is established in gpu, io op is needed
		 * where i is the start index in blob
		 */
		inline const void copy_data_io(vec_i &data_, int i)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
			memcpy(p_data(i),&data_[0],_cube_length*sizeof(unsigned int));
			cacu_memset(__BUFFER__, FAILEDHITS, _data_length);
		}

		/*
		 * copy data into blob, if blob is established in gpu, io op is needed
		 */
		inline const void copy_data_io(vec_i &data_)
		{
			CHECK_EQ_OP(data_.size(),_length,"blob size must be equal! %d vs %d",data_.size(),_length);
			memcpy(s_data(),&data_[0],_length*sizeof(unsigned int));
			cacu_memset(__BUFFER__, FAILEDHITS, _data_length);
		}

		/*
		 * copy data into blob, if blob is established in gpu, io op is needed
		 * where i is the start index in blob
		 */
		inline const void copy_diff_io(vec_t &data_, int i)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
			cacu_copy_cpu(&data_[0],_cube_length, p_diff(i));
			cacu_memset(__BUFFER__, FAILEDHITS, _data_length);
		}

		/*
		 * copy data into blob's diff, if blob is established in gpu, io op is needed
		 */
		inline const void copy_diff_io(vec_t &data_)
		{
			CHECK_EQ_OP(data_.size(),_length,"blob size must be equal! %d vs %d",data_.size(),_length);
			cacu_copy_cpu(&data_[0],_length, s_diff());
			cacu_memset(__BUFFER__, FAILEDHITS, _data_length);
		}

		virtual inline const int calculate_size() override{
			return test == _phrase ? _length * sizeof(unsigned int) : _length * sizeof(unsigned int) + _length * sizeof(float_t);
		}

		/**
		 * aysn data from device to host
		 */
		inline void _sync(int index_)
		{
			if(_num != 1){
				int b_index = index_ % _data_length;;

				if(__BUFFER__[b_index] != index_){
					LOG_FATAL("Miss index :%d in BUFFER!",index_);
				}
				else{
					device_copy2host(p_data(index_) , _d_p_data + b_index * _cube_length, _cube_length);
					if(train == _phrase)
						device_copy2host(p_diff(index_) , _d_p_diff + b_index * _cube_length, _cube_length);
				}
			}
		}

		/*
		 * serializa blob_bit data, output data to model file
		 * warning: bin_blob just serializa bit wise parameters.
		 */
		inline virtual const void serializa(std::ostream& os) override
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;
			bitset<32> _bits;

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
		}

		/*
		 * load blob_bit data from model file
		 */
		inline virtual const void load(std::ifstream& is) override
		{
			unsigned int* s_data_ = (unsigned int*)_s_data;

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
		}

		inline virtual const void resize(int num, int channel, int weight, int height) override
		{

		}

	protected:




	private:

		/**
		 * copy data into blob, if blob is established in gpu, io op is needed
		 * where i is the start index in blob
		 */
		inline int switch_host2dev(int i)
		{
			int buffer_index = i % _data_length;
			device_copy2dev(_d_p_data + buffer_index * _cube_length, p_data(i), _cube_length);
			if(train == _phrase)
				device_copy2dev(_d_p_diff + buffer_index * _cube_length, p_diff(i), _cube_length);
			return buffer_index;
		}


		//device use p_data
		unsigned int* _d_p_data;
		//device use p_diff
		float_t* _d_p_diff;

		int *__BUFFER__;

		int __HITS__(int index_)
		{
			int i = index_ % _data_length;
			if(__BUFFER__[i] == index_)
				return i;
			return FAILEDHITS;
		}

		int _data_length;

	};

};


