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

	class em_blob : public blob_base{


#define FAILEDHITS -1

	public:

		em_blob(int num, int channel, int width, int height, float_t _value, phrase_type phrase)
			:blob_base(num, channel, width, height, phrase, __em_blob__){

			_data_length = min(__EMBEDSIZE__,num);

			_s_data = (float_t*)malloc(_length * sizeof(float_t));
			set_data(_value);
			_d_p_data = device_malloc_v<float_t>(_data_length,_cube_length,_value);

			if (train == phrase){
				_s_diff = (float_t*)malloc(_length * sizeof(float_t));
				set_diff(0);
				_d_p_diff = device_malloc_v<float_t>(_data_length,_cube_length,0);
			}

			__BUFFER__ = (int *) malloc(_data_length * sizeof(int));
			cacu_memset(__BUFFER__, FAILEDHITS, _data_length);
		}

		~em_blob(){
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
		inline float_t* p_data(int n) {
			CHECK_LT_OP(n ,_num, "Index out of range %d vs %d!",n ,_num);
			return (float_t*)_s_data + n *_cube_length;
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
		inline float_t* s_data(){ return (float_t*)_s_data; }

		/**
		 * return the source probe in blob diff
		 */
		inline float_t* s_diff(){ return (float_t*)_s_diff; }


		inline float_t* s_data_d()
		{
			return _d_p_data;
		}

		inline float_t* s_diff_d()
		{
			return _d_p_diff;
		}

		inline float_t* p_data_d(int i)
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
			float_t* s_data_ = (float_t*)_s_data;
			float_t* s_diff_ = (float_t*)_s_diff;

			cacu_memset<mycnn::float_t>(s_data_, 0 ,_length);
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
		inline const void set_data(float_t value_)
		{
			float_t* s_data_ = (float_t*)_s_data;

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
		inline const void copy_data_io(vec_t &data_, int i)
		{
			CHECK_EQ_OP(data_.size(),_cube_length,"blob size must be equal! %d vs %d",data_.size(),_cube_length);
			cacu_copy_cpu(&data_[0],_cube_length, p_data(i));
			cacu_memset(__BUFFER__, FAILEDHITS, _data_length);
		}

		/*
		 * copy data into blob, if blob is established in gpu, io op is needed
		 */
		inline const void copy_data_io(vec_t &data_)
		{
			CHECK_EQ_OP(data_.size(),_length,"blob size must be equal! %d vs %d",data_.size(),_length);
			cacu_copy_cpu(&data_[0],_length, s_data());
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
			return test == _phrase ? _length * sizeof(float_t) : 2 * _length * sizeof(float_t);
		}

		/**
		 * aysn data from device to host
		 */
		inline void _sync(int index_)
		{
			if(_num != 1){
				int b_index = index_ % _data_length;

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
		 * serializa blob data, output data to model file
		 */
		inline virtual const void serializa(std::ostream& os) override
		{
			float_t* s_data_ = (float_t*)_s_data;

			os.write((char*)(&_length), sizeof(_length));
			for(int i = 0 ; i < _length; ++i)
				os.write((char*)(&s_data_[i]), sizeof(s_data_[i]));
		}

		/*
		 * loads blob data from model file
		 */
		inline virtual const void load(std::ifstream& is) override
		{
			float_t* s_data_ = (float_t*)_s_data;
			int length_;
			is.read(reinterpret_cast<char*>(&length_), sizeof(int));
			CHECK_EQ_OP(length_,_length,"parameter length is not equal to local weight: %d vs %d!",length_,_length);
			for (int i = 0; i < length_; i++){
				is.read(reinterpret_cast<char*>(s_data_ + i), sizeof(float_t));
			}
		}

	protected:




	private:

		/**
		 * copy data into blob, if blob is established in gpu, io op is needed
		 * where i is the start index in blob
		 */
		inline unsigned int switch_host2dev(int i)
		{
			int buffer_index = i % _data_length;
			device_copy2dev(_d_p_data + buffer_index * _cube_length, p_data(i), _cube_length);
			if(train == _phrase)
				device_copy2dev(_d_p_diff + buffer_index * _cube_length, p_diff(i), _cube_length);
			return buffer_index;
		}

		//device use p_data
		float_t* _d_p_data;
		//device use p_diff
		float_t* _d_p_diff;

		int *__BUFFER__;

		inline int __HITS__(int index_)
		{
			int i = index_ % _data_length;
			if(__BUFFER__[i] == index_)
				return i;
			return FAILEDHITS;
		}

		int _data_length;

	};


};

