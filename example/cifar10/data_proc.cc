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


#include "data_proc.h"

#include "../../cacu/utils/log.h"
#include "../../cacu/math/math_functions.h"

using namespace cacu;


void readdata(chars_t filename, vector<vec_t> &data_blob) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if(!data_file)
		LOG_FATAL("file %s cannot be opened!",filename.c_str());
	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			datas[j] = (cacu::float_t) ((unsigned char) (buffer[j]));
			datas[j + kCIFARDataSize] = (cacu::float_t) ((unsigned char) (buffer[j + kCIFARDataSize]));
			datas[j + kCIFARDataSize * 2] = (cacu::float_t) ((unsigned char) (buffer[j + 2 * kCIFARDataSize]));
		}
		data_blob.push_back(datas);
	}
}

void readdata(chars_t filename, vector<vec_t> &data_blob,vec_t &mean) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if(!data_file)
		LOG_FATAL("file %s cannot be opened!",filename.c_str());
	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			datas[j] = (cacu::float_t) ((unsigned char) (buffer[j])) - mean[j];
			datas[j + kCIFARDataSize] = (cacu::float_t) ((unsigned char) (buffer[j + kCIFARDataSize])) - mean[j + kCIFARDataSize];
			datas[j + kCIFARDataSize * 2] = (cacu::float_t) ((unsigned char) (buffer[j + 2 * kCIFARDataSize])) - mean[j + 2 * kCIFARDataSize];
		}
		data_blob.push_back(datas);
	}
}

void readdata(string filename, vector<vec_t> &data_blob,
	vector<vec_i> &labels) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if (!data_file)
		LOG_FATAL("file %s cannot be opened!", filename.c_str());
	cacu::float_t *snp;
	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		labels.push_back(vec_i(1, (unsigned int)((label_char))));
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			datas[j] = ((cacu::float_t) ((unsigned char)(buffer[j]))) / 255.0;
			datas[j + kCIFARDataSize] = ((cacu::float_t) ((unsigned char)(buffer[j + kCIFARDataSize])))/ 255.0;
			datas[j + kCIFARDataSize * 2] = ((cacu::float_t) ((unsigned char)(buffer[j + 2 * kCIFARDataSize])))/ 255.0;
		}
		data_blob.push_back(datas);
	}
}

void readdata(string filename, vector<vec_t> &data_blob, vec_t &mean,
		vector<vec_i> &labels) {
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if(!data_file)
		LOG_FATAL("file %s cannot be opened!",filename.c_str());
	cacu::float_t *snp;
	for (unsigned int i = 0; i < kCIFARBatchSize; i++) {
		char label_char;
		data_file.read(&label_char, 1);
		labels.push_back(vec_i(1, (unsigned int)((label_char))));
		char buffer[kCIFARImageNBytes];
		data_file.read(buffer, kCIFARImageNBytes);
		vec_t datas(kCIFARImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kCIFARDataSize; j++) {
			datas[j] = ((cacu::float_t) ((unsigned char) (buffer[j])) - mean[j]);
			datas[j + kCIFARDataSize] = ((cacu::float_t) ((unsigned char) (buffer[j + kCIFARDataSize])) - mean[j + kCIFARDataSize]);
			datas[j + kCIFARDataSize * 2] = ((cacu::float_t) ((unsigned char) (buffer[j + 2 * kCIFARDataSize])) - mean[j + 2 * kCIFARDataSize]);
		}
		data_blob.push_back(datas);
	}
}

void load_data_bymean(string filepath, string meanfile, vector<vec_t> &data_blob, vector<vec_i> &labels)
{

	vec_t mean(kCIFARImageNBytes);

	imageio_utils::load_mean_file(&mean[0], meanfile);

	for (int i = 1; i <= 5; i++) {
		ostringstream oss;
		oss << filepath << "data_batch_" << i << ".bin";
		readdata((oss.str()), data_blob , mean , labels);
	}
}

void load_data(string filepath, vector<vec_t> &data_blob, vector<vec_i> &labels)
{

	for (int i = 1; i <= 5; i++) {
		ostringstream oss;
		oss << filepath << "data_batch_" << i << ".bin";
		readdata((oss.str()), data_blob, labels);
	}
}

void load_test_data_bymean(string filepath, string meanfile, vector<vec_t> &data_blob, vector<vec_i> &labels)
{

	vec_t mean(kCIFARImageNBytes);

	imageio_utils::load_mean_file(&mean[0], meanfile);

	{
		ostringstream oss;
		oss << filepath << "test_batch.bin";
		readdata((oss.str()), data_blob , mean, labels);
	}
}

void load_test_data(string filepath, vector<vec_t> &data_blob, vector<vec_i> &labels)
{

	{
		ostringstream oss;
		oss << filepath << "test_batch.bin";
		readdata((oss.str()), data_blob, labels);
	}
}

vec_t compute_mean(chars_t &filepath, int filecount)
{
	vector<vec_t> mean_data;
	vec_t mean(kCIFARImageNBytes);

	//calculate mean
	for (int i = 1; i <= filecount; i++) {
		ostringstream oss;
		oss << filepath << "data_batch_" << i << ".bin";
		readdata((oss.str()), mean_data);
	}

	cacu::float_t length = (cacu::float_t) mean_data.size();

	for (unsigned int i = 0; i < mean_data.size(); i++) {
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_saxpby_oblas(&mean_data[i][0], 1, &mean[0], 1, kCIFARImageNBytes);
#elif __CBLASTYPE__ == __MKL__
	cacu_saxpby_mkl(&mean_data[i][0], 1, &mean[0], 1, kCIFARImageNBytes);
#endif
	}
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_scalex_oblas(&mean[0], (float)1.0/length, kCIFARImageNBytes);
#elif __CBLASTYPE__ == __MKL__
	cacu_scalex_mkl(&mean[0], (float)1.0/length, kCIFARImageNBytes);
#endif
	return mean;
}


void make_mean(chars_t filepath, chars_t meanfile)
{
	vec_t mean = compute_mean(filepath,5);
	//LOG_DEBUG("%d", mean.size());
	imageio_utils::save_mean_file(&mean[0],meanfile,mean.size());
}




