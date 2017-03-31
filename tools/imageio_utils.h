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

#ifdef _WIN32
#include <iostream>
#include <string>
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
using namespace Gdiplus;

#elif linux

#include <string>
#include <libpng/png.h>
#include <stdlib.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace cv;

#endif

#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "../core/math/cuda/cuda_utils.h"


using namespace std;


namespace mycnn_tools{


	class imageio_utils {

	public:

		static void imread_gpu(float *p_data,string file_path_)
		{
			cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
			unsigned int height = src.rows;
			unsigned int width = src.cols;
			unsigned int c_length = height * width;

			vec_t tmp_(3*c_length);
			unsigned int index;
			for (unsigned int y = 0; y < height; y++)
				for (unsigned int x = 0; x < width; x++) {
					index = y * width + x;
					tmp_[index] = ((float) src.at<cv::Vec3b>(y, x)[0]);
					tmp_[c_length + index] = ((float) src.at<cv::Vec3b>(y, x)[1]);
					tmp_[2*c_length + index] = ((float) src.at<cv::Vec3b>(y, x)[2]);
				}
			cuda_copy2dev(p_data,&tmp_[0],tmp_.size());
			vec_t().swap(tmp_);
		}

		static void imread(float *p_data,string file_path_)
		{
			cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
			unsigned int height = src.rows;
			unsigned int width = src.cols;
			unsigned int c_length = height * width;

			unsigned int index;
			for (unsigned int y = 0; y < height; y++)
				for (unsigned int x = 0; x < width; x++) {
					index = y * width + x;
					p_data[index] = ((float) src.at<cv::Vec3b>(y, x)[0]);
					p_data[c_length + index] = ((float) src.at<cv::Vec3b>(y, x)[1]);
					p_data[2*c_length + index] = ((float) src.at<cv::Vec3b>(y, x)[2]);
				}
		}

		static void save_mean_file(float *p_data, string mean_file_ , int length_)
		{
			ofstream os(mean_file_, ios::binary);
			os.precision(numeric_limits<float>::digits10);
			if(!os)
				LOG_FATAL("file %s cannot be opened!",mean_file_.c_str());
			for(int i = 0 ; i < length_; ++i)
			{
				os.write((char*)(p_data+i), sizeof(float));
			}
			os.close();
		}

		static void load_mean_file(float *p_data, string mean_file_)
		{
			ifstream is(mean_file_);
			is.precision(numeric_limits<float>::digits10);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",mean_file_.c_str());
			vector<float> temp_;
			float fp_;

			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(float));
				temp_.push_back(fp_);
			}
			is.close();

			float *d_= &temp_[0];
			memcpy(p_data,d_,temp_.size()*sizeof(float));
		}

		static void load_mean_file_gpu(float *p_data, string mean_file_)
		{
			ifstream is(mean_file_);
			is.precision(numeric_limits<float>::digits10);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",mean_file_.c_str());
			vector<float> temp_;
			float fp_;

			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(float));
				temp_.push_back(fp_);
			}
			is.close();

			float *d_= &temp_[0];
			cuda_copy2dev(p_data,d_,temp_.size());
		}

	private:



	};
}
