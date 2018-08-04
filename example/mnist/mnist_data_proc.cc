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


#include "mnist_data_proc.h"

#include "../../cacu/utils/log.h"
#include "../../cacu/math/math_functions.h"

using namespace std;
using namespace cacu;

unsigned int swap_endian(unsigned int val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}


void readdata_mnist(chars_t filename, vector<vec_t> &data_blob) {
	char label_char;
	unsigned int data_size_;
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if (!data_file)
		LOG_FATAL("file %s cannot be opened!", filename.c_str());
	cout << "magic number: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  number of images: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  image size: (";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ",";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ")" << endl;

	cacu::float_t *snp;
	for (unsigned int i = 0; i < kMNISTBatchSize; i++) {
		
		char buffer[kMNISTImageNBytes];
		data_file.read(buffer, kMNISTImageNBytes);
		vec_t datas(kMNISTImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kMNISTDataSize; j++) {
			datas[j] = ((cacu::float_t) ((unsigned char) (buffer[j])));
		}
		data_blob.push_back(datas);
	}
}

void readdata_mnist(chars_t filename, vector<vec_t> &data_blob,vec_t &mean) {
	char label_char;
	
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if (!data_file)
		LOG_FATAL("file %s cannot be opened!", filename.c_str());
	unsigned int data_size_;
	cout << "magic number: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  number of images: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  image size: (";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ",";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ")" << endl;
	
	cacu::float_t *snp;
	for (unsigned int i = 0; i < kMNISTBatchSize; i++) {
		
		char buffer[kMNISTImageNBytes];
		data_file.read(buffer, kMNISTImageNBytes);
		vec_t datas(kMNISTImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kMNISTDataSize; j++) {
			datas[j] = ((cacu::float_t) ((unsigned char) (buffer[j])) - mean[j]);
		}
		data_blob.push_back(datas);
	}
}

void readdata_mnist(string filename, string labelname, vector<vec_t> &data_blob,
	vector<vec_i> &labels) {
		
	char label_char;
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if (!data_file)
		LOG_FATAL("file %s cannot be opened!", filename.c_str());
	unsigned int data_size_;
	cout << "magic number: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  number of images: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  image size: (";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ",";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ")" << endl;

	std::ifstream label_file(labelname, std::ios::in | std::ios::binary);
	if (!label_file)
		LOG_FATAL("file %s cannot be opened!", labelname.c_str());
	cout << "magic number: ";
	label_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  number of items: ";
	label_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << endl;
	
	cacu::float_t *snp;
	for (unsigned int i = 0; i < kMNISTBatchSize; i++) {
		
		label_file.read(&label_char, 1);
		labels.push_back(vec_i(1, (unsigned int)((label_char))));
		char buffer[kMNISTImageNBytes];
		data_file.read(buffer, kMNISTImageNBytes);
		vec_t datas(kMNISTImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kMNISTDataSize; j++) {
			datas[j] = ((cacu::float_t) ((unsigned char)(buffer[j])));
		}
		data_blob.push_back(datas);
	}
}

void readdata_mnist(string filename, string labelname, vector<vec_t> &data_blob, vec_t &mean,
		vector<vec_i> &labels) {
	char label_char;
	std::ifstream data_file(filename, std::ios::in | std::ios::binary);
	if (!data_file)
		LOG_FATAL("file %s cannot be opened!", filename.c_str());
	unsigned int data_size_;
	cout << "magic number: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  number of images: ";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  image size: (";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ",";
	data_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << ")" << endl;

	std::ifstream label_file(labelname, std::ios::in | std::ios::binary);
	if (!label_file)
		LOG_FATAL("file %s cannot be opened!", labelname.c_str());
	cout << "magic number: ";
	label_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << "  number of items: ";
	label_file.read(reinterpret_cast<char*>(&data_size_), sizeof(unsigned int));
	cout << swap_endian(data_size_) << endl;
	
	cacu::float_t *snp;
	for (unsigned int i = 0; i < kMNISTBatchSize; i++) {
		
		label_file.read(&label_char, 1);
		labels.push_back(vec_i(1, (unsigned int)((label_char))));
		char buffer[kMNISTImageNBytes];
		data_file.read(buffer, kMNISTImageNBytes);
		vec_t datas(kMNISTImageNBytes);
		snp = &datas[0];
		for (unsigned int j = 0; j < kMNISTDataSize; j++) {
			datas[j] = ((cacu::float_t) ((unsigned char) (buffer[j])) - mean[j]);
		}
		data_blob.push_back(datas);
	}
}

void load_data_bymean_mnist(string filepath, string meanfile, vector<vec_t> &data_blob, vector<vec_i> &labels)
{
	LOG_INFO("Reading mnist data...");
	vec_t mean(kMNISTImageNBytes);

	imageio_utils::load_mean_file(&mean[0], meanfile);

	ostringstream oss;
	oss << filepath << "train-images.idx3-ubyte";
	
	ostringstream label_oss;
	label_oss << filepath << "train-labels.idx1-ubyte";
	
	readdata_mnist((oss.str()), label_oss.str(), data_blob , mean , labels);
	
}

void load_data_mnist(string filepath, vector<vec_t> &data_blob, vector<vec_i> &labels)
{
	ostringstream oss;
	oss << filepath << "train-images.idx3-ubyte";
	
	ostringstream label_oss;
	label_oss << filepath << "train-labels.idx1-ubyte";
	
	readdata_mnist((oss.str()), label_oss.str(), data_blob, labels);
	
}

void load_test_data_bymean_mnist(string filepath, string meanfile, vector<vec_t> &data_blob, vector<vec_i> &labels)
{

	vec_t mean(kMNISTImageNBytes);

	imageio_utils::load_mean_file(&mean[0], meanfile);

	ostringstream oss;
	oss << filepath << "t10k-images.idx3-ubyte";
	
	ostringstream label_oss;
	label_oss << filepath << "t10k-labels.idx1-ubyte";
	
	readdata_mnist((oss.str()), label_oss.str(), data_blob , mean , labels);
}

void load_test_data_mnist(string filepath, vector<vec_t> &data_blob, vector<vec_i> &labels)
{

	ostringstream oss;
	oss << filepath << "t10k-images.idx3-ubyte";
	
	ostringstream label_oss;
	label_oss << filepath << "t10k-labels.idx1-ubyte";
	
	readdata_mnist((oss.str()), label_oss.str(), data_blob, labels);
	
}

vec_t compute_mean_mnist(chars_t &filepath, int filecount)
{
	vector<vec_t> mean_data;
	vec_t mean(kMNISTImageNBytes);

	//calculate mean
	for (int i = 1; i <= filecount; i++) {
		ostringstream oss;
		oss << filepath << "train-images.idx3-ubyte";
		readdata_mnist((oss.str()), mean_data);
	}

	cacu::float_t length = (cacu::float_t) mean_data.size();

	for (unsigned int i = 0; i < mean_data.size(); i++) {
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_saxpby_oblas(&mean_data[i][0], 1, &mean[0], 1, kMNISTImageNBytes);
#elif __CBLASTYPE__ == __MKL__
	cacu_saxpby_mkl(&mean_data[i][0], 1, &mean[0], 1, kMNISTImageNBytes);
#endif
	}
#if __CBLASTYPE__ == __OPENBLAS__
	cacu_scalex_oblas(&mean[0], (float)1.0/length, kMNISTImageNBytes);
#elif __CBLASTYPE__ == __MKL__
	cacu_scalex_mkl(&mean[0], (float)1.0/length, kMNISTImageNBytes);
#endif
	return mean;
}


void make_mean_mnist(chars_t filepath, chars_t meanfile)
{
	vec_t mean = compute_mean_mnist(filepath,1);
	//LOG_DEBUG("%d", mean.size());
	imageio_utils::save_mean_file(&mean[0],meanfile,mean.size());
}




