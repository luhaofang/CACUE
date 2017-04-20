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

#ifndef _WIN32

#include <vector>
#include <stdarg.h>
#include <pthread.h>
#include <unistd.h>

#if __PARALLELTYPE__ == __CUDA__

#include <cuda_runtime_api.h>

#include "../imageio_utils.h"

using namespace std;
using namespace mycnn_tools;

namespace mycnn{

#define MAX_BUFF_SIZE 10

/**
 * Thread condition identifier type
 */
typedef enum thread_state {forked, not_forked, terminated} asyn_type;

/**
 * Buffer container for asynchronous control
 */
typedef struct buffer_meta{
	float_t *s_data;
	unsigned int *s_label;
	asyn_type is_forked;
} buff_item;

/**
 * mutex for global parameter iteration
 */
pthread_mutex_t itermutex=PTHREAD_MUTEX_INITIALIZER;

/**
 * data_blob probe index
 */
int _asyn_index = 0;

/**
 * mutex iteration
 */
int _asyn_iter = 0;

/**
 * thread index for pthread creator
 */
int _thread_index = 0;

/**
 * threads
 */
pthread_t *_threads;

/**
 * feeding buffer
 */
vector<buff_item *> _asyn_buff;

/*
 * probe for global data blob
 */
vector<string> *_asyn_data_blob;

/**
 * probe for global label blob
 */
vector<vec_i> *_asyn_data_label;

int _asyn_batch_size;

int _asyn_max_iter;

/**
 * buffer image count
 */
int _asyn_num;

/**
 * buffer length for each image
 */
int _asyn_length;

/**
 * probe for image mean data
 */
float_t *_asyn_mean;

void *asyn_fork(void *args)
{
	int thread_id = (int)(*((int*)args));
	if(thread_id < MAX_BUFF_SIZE)
	{
		buff_item *buff = _asyn_buff[thread_id];
		vector<string> batch_blob(_asyn_batch_size);
		vec_i batch_label(_asyn_batch_size);
		while(_asyn_iter < _asyn_max_iter)
		{
			if(buff->is_forked == forked)
			{
				pthread_mutex_lock(&itermutex);
				_asyn_iter += 1;
				for(int i = 0 ; i < _asyn_batch_size ; ++i)
				{
					if(_asyn_index >= _asyn_data_blob->size())
						_asyn_index = 0;
					batch_blob[i] = _asyn_data_blob->at(_asyn_index);
					batch_label[i] = _asyn_data_label->at(_asyn_index)[0];
					_asyn_index += 1;
				}
				pthread_mutex_unlock(&itermutex);
				for(int i = 0 ; i < _asyn_batch_size ; ++i)
				{
					imageio_utils::imread(buff->s_data + i * _asyn_length,batch_blob[i]);
					cacu_saxpy_cpu(_asyn_mean,(mycnn::float_t)-1,buff->s_data + i * _asyn_length, _asyn_length);
					memcpy(buff->s_label,&batch_label[0],_asyn_batch_size*sizeof(unsigned int));
				}
				buff->is_forked = not_forked;
			}
			usleep(100);
		}
		buff->is_forked = terminated;
	}
	return NULL;
}

void *asyn_fork_without_mean(void *args)
{
	int thread_id = (int)(*((int*)args));
	if(thread_id < MAX_BUFF_SIZE)
	{
		buff_item *buff = _asyn_buff[thread_id];
		vector<string> batch_blob(_asyn_batch_size);
		vec_i batch_label(_asyn_batch_size);
		while(_asyn_iter < _asyn_max_iter)
		{
			if(buff->is_forked == forked)
			{
				pthread_mutex_lock(&itermutex);
				_asyn_iter += 1;
				for(int i = 0 ; i < _asyn_batch_size ; ++i)
				{
					if(_asyn_index >= _asyn_data_blob->size())
						_asyn_index = 0;
					batch_blob[i] = _asyn_data_blob->at(_asyn_index);
					batch_label[i] = _asyn_data_label->at(_asyn_index)[0];
					_asyn_index += 1;
				}
				pthread_mutex_unlock(&itermutex);
				for(int i = 0 ; i < _asyn_batch_size ; ++i)
				{
					imageio_utils::imread(buff->s_data + i * _asyn_length,batch_blob[i]);
					//cacu_saxpy_cpu(_asyn_mean,(mycnn::float_t)-1,buff->s_data + i * _asyn_length, _asyn_length);
					memcpy(buff->s_label,&batch_label[0],_asyn_batch_size*sizeof(unsigned int));
				}
				buff->is_forked = not_forked;
			}
			usleep(100);
		}
		buff->is_forked = terminated;
	}
	return NULL;
}

void asyn_initial(int num,int length,int max_iter, vector<string> *data_blob,vector<vec_i> *data_label, float_t *mean){
	_asyn_buff = vector<buff_item *>(MAX_BUFF_SIZE);
	//initial buffer source
	for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
	{
		buff_item *bi = new buff_item();
		bi->is_forked = forked;
		bi->s_data = (float_t*) malloc(num * length * sizeof(float_t));
		bi->s_label = (unsigned int*) malloc(num * sizeof(unsigned int));
		_asyn_buff[i] = bi;
	}
	_threads = new pthread_t[MAX_BUFF_SIZE];

	_asyn_num = num;
	_asyn_length = length;
	_asyn_batch_size = num;
	_asyn_max_iter = max_iter;
	_asyn_data_blob = data_blob;
	_asyn_data_label = data_label;
	_asyn_mean = mean;
};

void asyn_release(){

	for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
	{
		free(_asyn_buff[i]->s_data);
		free(_asyn_buff[i]->s_label);
	}
	vector<buff_item *>().swap(_asyn_buff);
	delete _threads;
};

void asyn_initial_threads()
{
	int rc;
	_thread_index = MAX_BUFF_SIZE;
	while(_thread_index > 0)
	{
		_thread_index--;
		rc = pthread_create(&_threads[_thread_index], NULL, asyn_fork_without_mean, (void*)&_thread_index);
		if (rc)
		{
			printf("ERROR; return code is %d\n", rc);
			exit(-1);
		}
	}
}

inline bool _ASYN_TERMINATED()
{
	for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
	{
		buff_item *buff = _asyn_buff[i];
		if(buff->is_forked != terminated)
			return false;
	}
	return true;
}

void asyn_get_gpu_data(float_t *data_,unsigned int *label_)
{
	while(true){
		for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
		{
			buff_item *buff = _asyn_buff[i];
			if(buff->is_forked == not_forked)
			{
#if __USDYNAMIC__ == ON
				memcpy(data_, buff->s_data, _asyn_length * _asyn_num * sizeof(float_t));
				memcpy(label_, buff->s_label, _asyn_num * sizeof(unsigned int));
#else
				res = cudaMemcpy(data_, buff->s_data, _asyn_length * _asyn_num * sizeof(float_t),cudaMemcpyHostToDevice);
				CUDA_CHECK(res);
				res = cudaMemcpy(label_, buff->s_label, _asyn_num * sizeof(unsigned int),cudaMemcpyHostToDevice);
				CUDA_CHECK(res);
#endif
				buff->is_forked = forked;
				return;
			}
		}
		if(_ASYN_TERMINATED())
			return;
		usleep(100);
	}
}

void asyn_join_threads()
{
	for(int t = 0; t < MAX_BUFF_SIZE; t++)
		pthread_join(_threads[t],NULL);
}


};

#endif

#endif
