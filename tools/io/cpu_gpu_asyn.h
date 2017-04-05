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

#include <vector>
#include <stdarg.h>
#include <pthread.h>

#include <cuda_runtime_api.h>

#include "../imageio_utils.h"

using namespace std;
using namespace mycnn_tools;

namespace mycnn{

	#define MAX_BUFF_SIZE 10

	typedef enum thread_state {forked, not_forked, terminated} asyn_type;

	typedef struct buffer_meta{
		float_t *s_data;
		unsigned int *s_label;
		asyn_type is_forked;
	} buff_item;

	class cpu_gpu_asyn {

	public:

		pthread_mutex_t itermutex=PTHREAD_MUTEX_INITIALIZER;

		cpu_gpu_asyn(int num,int length,int batch_size,int max_iter, vector<string> *&data_blob,vec_i *&data_label, float_t *&mean){
			_buff = vector<buff_item *>(MAX_BUFF_SIZE);
			cudaError_t res;
			//initial buffer source
			for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
			{
				buff_item *bi = new buff_item();
				bi->is_forked = forked;
				res = cudaMalloc((void**) (&bi->s_data), num * length * sizeof(float_t));
				CUDA_CHECK(res);
				res = cudaMalloc((void**) (&bi->s_label), num * sizeof(unsigned int));
				CUDA_CHECK(res);
				_buff.push_back(bi);
			}
			_threads = new pthread_t[NUM_THREADS];

			_num = num;
			_length = length;
			_batch_size = batch_size;
			_max_iter = max_iter;
			_data_blob = data_blob;
			_data_label = data_label;
			_mean = mean;
		};

		~cpu_gpu_asyn(){

			for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
			{
				cudaFree(_buff[i]->s_data);
				cudaFree(_buff[i]->s_label);
			}
			vector<buff_item *>().swap(_buff);
			delete _threads;
		};

		void initial_threads()
		{
			int rc,i;
			for(i = 0 ; i < MAX_BUFF_SIZE; ++i)
			{
				rc = pthread_create(&_threads[i], NULL, fork, &i);
				if (rc)
				{
					printf("ERROR; return code is %d\n", rc);
					exit(-1);
				}
			}
		}

		void *fork(int max_iter,int thread_id,int batch_size)
		{
			buff_item *buff = _buff[thread_id];
			vector<string> batch_blob(batch_size);
			vec_i batch_lable(batch_size);
			while(_iter < max_iter)
			{
				if(buff->is_forked == forked)
				{
					pthread_mutex_lock(&itermutex);
					_iter += 1;
					for(int i = 0 ; i < _batch_size ; ++i)
					{
						if(index >= _data_blob->size())
							index = 0;
						batch_blob[i] = _data_blob[index];
						buff->s_label[i] = _data_label[index];
						index += 1;
					}
					pthread_mutex_unlock(&itermutex);
					for(int i = 0 ; i < _batch_size ; ++i)
					{
						imageio_utils::imread_gpu(buff->s_data + i * _length,batch_blob[i]);
						cacu_saxpy(_mean,(mycnn::float_t)-1,buff->s_data + i * _length, _length);
					}
					buff->is_forked = not_forked;
				}
				sleep(1000);
			}
			buff->is_forked = terminated;
			return NULL;
		}

		void get_gpu_data(float_t *data_,bin_blob *label_)
		{
			while(true){
				for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
				{
					buff_item *buff = _buff[i];
					if(buff->is_forked == not_forked)
					{
						cudaMemcpy(data_, buff->s_data, _length * _num * sizeof(float_t),cudaMemcpyDeviceToDevice);
						cudaMemcpy(label_, buff->s_label, _num * sizeof(unsigned int),cudaMemcpyDeviceToDevice);
						buff->is_forked = forked;
					}
				}
				if(_TERMINATED())
					return;
				sleep(1000);
			}
		}


		int _index = 0;

		int _iter = 0;

	private:

		pthread_t *_threads;

		vector<buff_item *> _buff;

		vector<string> *_data_blob;

		vec_i *_data_label;

		int _batch_size;

		int _max_iter;

		int _num;

		int _length;

		float_t *_mean;

		inline bool _TERMINATED()
		{
			for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
			{
				buff_item *buff = _buff[i];
				if(buff->is_forked != terminated)
					return false;
			}
			return true;
		}

	};

};
