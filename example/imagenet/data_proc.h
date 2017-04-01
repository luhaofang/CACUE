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


#include <time.h>

#include "../../mycnn.h"

#include "../../tools/imageio_utils.h"


using namespace mycnn;
using namespace mycnn_tools;


const int KIMAGESIZE = 3 * 227 * 227;


void readdata(chars_t filename, mycnn::float_t *data_) {
#if __PARALLELTYPE__ == __GPU__
	imageio_utils::imread_gpu(data_,filename);
#else
	imageio_utils::imread(data_,filename);
#endif
}

void readdata(chars_t filename, mycnn::float_t *data_,mycnn::float_t *mean_) {
#if __PARALLELTYPE__ == __GPU__
	imageio_utils::imread_gpu(data_,filename);
#else
	imageio_utils::imread(data_,filename);
#endif
	cacu_saxpy(mean_,(mycnn::float_t)-1,data_,KIMAGESIZE);
}

vec_t compute_mean(chars_t &filepath, chars_t &filelist)
{
	vec_t mean(KIMAGESIZE);
	vec_t temp(KIMAGESIZE);
	string file_;
	ifstream is(filelist);
	is.precision(numeric_limits<float>::digits10);
	if(!is)
		LOG_FATAL("file %s cannot be opened!",filelist.c_str());
	int count = 0;
	while( getline(is,file_) )
	{
		imageio_utils::imread(&temp[0],filepath + file_);
		cacu_saxpy_oblas(&temp[0], 1, &mean[0],KIMAGESIZE);
		count += 1;
		if(count % 1000 == 0)
			LOG_INFO("make mean file process : %d",count);
	}
	cacu_scalex_oblas(&mean[0],(float)1.0/count,KIMAGESIZE);
	return mean;
}


void make_mean(chars_t filepath, chars_t filelist, chars_t meanfile)
{
	LOG_INFO("start making mean file ... ");
	vec_t mean = compute_mean(filepath,filelist);
	LOG_DEBUG("%f,%f",mean[0],mean[24]);
	imageio_utils::save_mean_file(&mean[0],meanfile,mean.size());
}




