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
#ifndef DATA_PROC_H_
#define DATA_PROC_H_

#include <time.h>

#include "../../cacu/cacu.h"

#include "../../tools/imageio_utils.hpp"


using namespace cacu;
using namespace cacu_tools;
using namespace std;


const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARDataSize = 1024;
const int kCIFARDataCount = 50000;

void readdata(chars_t filename, vector<vec_t> &data_blob);

void readdata(chars_t filename, cacu::float_t *data_);

void readdata(chars_t filename, vector<vec_t> &data_blob,vec_t &mean);

void readdata(string filename, vector<vec_t> &data_blob, vector<vec_i> &labels);

void readdata(string filename, vector<vec_t> &data_blob, vec_t &mean,
		vector<vec_i> &labels);

void load_data(string filepath, vector<vec_t> &data_blob, vector<vec_i> &labels);

void load_data_bymean(string filepath, string meanfile, vector<vec_t> &data_blob, vector<vec_i> &labels);

void load_test_data(string filepath, vector<vec_t> &data_blob, vector<vec_i> &labels);

void load_test_data_bymean(string filepath, string meanfile, vector<vec_t> &data_blob, vector<vec_i> &labels);

vec_t compute_mean(chars_t &filepath, int filecount);

void make_mean(chars_t filepath, chars_t meanfile);





#endif