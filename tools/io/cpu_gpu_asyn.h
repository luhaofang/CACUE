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

#include <cuda_runtime.h>

using namespace std;

namespace mycnn{

#define MAX_BUFF_SIZE 10

	class cpu_gpu_asyn {

	public:

		cpu_gpu_asyn(){
			_buff = vector<buff_item *>(MAX_BUFF_SIZE);
			//initial buffer source
			for(int i = 0 ; i < MAX_BUFF_SIZE; ++i)
			{
				buff_item *bi = new buff_item();
				bi->is_forked = false;
				bi->s_data = NULL;
				_buff.push_back(bi);
			}
			int thread_count = pthread;
			_threads.push_back();
		};

		~cpu_gpu_asyn(){

			delete _buff;
		};

		void thread_pool()
		{

		}

		void fork()
		{

		}



	private:

		vector<pthread_t> _threads;

		typedef struct buffer_meta{
			float_t *s_data;
			bool is_forked;
		} buff_item;

		vector<buff_item *> _buff;

	};
}
