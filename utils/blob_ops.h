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

#include <ostream>
#include <fstream>
#include <sstream>


using namespace std;

namespace mycnn {

	class blob_ops{

	public:

		static void read_data2blob(blob *b, string filepath)
		{
			vec_t data_;
			ifstream is(filepath);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",filepath);
			is.precision(numeric_limits<float_t>::digits10);
			float_t fp_;
			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(float_t));
				data_.push_back(fp_);
			}
			is.close();

			b->copy_data_io(data_);
		}

		static void read_data2blob(blob *b, string filepath,int index)
		{
			vec_t data_;
			ifstream is(filepath);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",filepath);
			is.precision(numeric_limits<float_t>::digits10);
			float_t fp_;
			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(float_t));
				data_.push_back(fp_);
			}
			is.close();

			b->copy_data_io(data_,index);
		}

		static void read_diff2blob(blob *b, string filepath)
		{
			vec_t data_;
			ifstream is(filepath);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",filepath);
			is.precision(numeric_limits<float_t>::digits10);
			float_t fp_;
			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(float_t));
				data_.push_back(fp_);
			}
			is.close();

			b->copy_diff_io(data_);
		}

		static void read_diff2blob(blob *b, string filepath,int index)
		{
			vec_t data_;
			ifstream is(filepath);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",filepath);
			is.precision(numeric_limits<float_t>::digits10);
			float_t fp_;
			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(float_t));
				data_.push_back(fp_);
			}
			is.close();

			b->copy_diff_io(data_,index);
		}
	};

};
