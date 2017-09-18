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
#include <opencv2/opencv.hpp>
using namespace cv;

#endif

#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "../core/math/cuda/cuda_utils.h"
#include "../core/utils/data_defination.h"

using namespace std;


namespace mycnn_tools{


	class imageio_utils {

	public:
#ifdef _WIN32
#if __PARALLELTYPE__ == __CUDA__
		static void imread_gpu(mycnn::float_t *p_data, string file_path_)
		{
			vec_t temp_(3 * c_length);
			GdiplusStartupInput gdiplusstartupinput;
			ULONG_PTR gdiplustoken;
			GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);
			Bitmap* bmp = new Bitmap(StringToWString(file_path_).c_str());
			unsigned int height = bmp->GetHeight();
			unsigned int width = bmp->GetWidth();
			Color color;
			unsigned int c_length = height * width;
			unsigned int index;
			for (unsigned int y = 0; y < height; y++)
				for (unsigned int x = 0; x < width; x++)
				{
					index = y * width + x;
					bmp->GetPixel(x, y, &color);
					temp_[index] = ((mycnn::float_t)color.GetRed());
					temp_[c_length + index] = ((mycnn::float_t)color.GetGreen());
					temp_[2 * c_length + index] = ((mycnn::float_t)color.GetBlue());
				}
			delete bmp;
			GdiplusShutdown(gdiplustoken);
			cuda_copy2dev(p_data, &temp_[0], temp_.size());
			vec_t().swap(temp_);
		}
#endif
		static void imread(mycnn::float_t *p_data, string file_path_)
		{
			GdiplusStartupInput gdiplusstartupinput;
			ULONG_PTR gdiplustoken;
			GdiplusStartup(&gdiplustoken, &gdiplusstartupinput, NULL);
			Bitmap* bmp = new Bitmap(StringToWString(file_path_).c_str());
			unsigned int height = bmp->GetHeight();
			unsigned int width = bmp->GetWidth();
			Color color;
			unsigned int c_length = height * width;
			unsigned int index;
			for (unsigned int y = 0; y < height; y++)
				for (unsigned int x = 0; x < width; x++)
				{
					index = y * width + x;
					bmp->GetPixel(x, y, &color);
					p_data[index] = ((mycnn::float_t)color.GetRed());
					p_data[c_length + index] = ((mycnn::float_t)color.GetGreen());
					p_data[2 * c_length + index] = ((mycnn::float_t)color.GetBlue());
				}
			delete bmp;
			GdiplusShutdown(gdiplustoken);
		}
#else

#if __PARALLELTYPE__ == __CUDA__
		static void imread_gpu(mycnn::float_t *p_data,const char* file_path_)
		{
			cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
			if(!src.data)
				LOG_FATAL("file %s cannot be opened!",file_path_);

			unsigned int height = src.rows;
			unsigned int width = src.cols;
			unsigned int c_length = height * width;

			vec_t temp_(3*c_length);
			unsigned int index;
			for (unsigned int y = 0; y < height; y++)
				for (unsigned int x = 0; x < width; x++) {
					index = y * width + x;
					temp_[index] = ((mycnn::float_t) src.at<cv::Vec3b>(y, x)[0]);
					temp_[c_length + index] = ((mycnn::float_t) src.at<cv::Vec3b>(y, x)[1]);
					temp_[2*c_length + index] = ((mycnn::float_t) src.at<cv::Vec3b>(y, x)[2]);
				}

			cuda_copy2dev(p_data,&temp_[0],temp_.size());
			vec_t().swap(temp_);
		}
#endif
		static void imread(mycnn::float_t *p_data,const char* file_path_)
		{
			cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
			if(!src.data)
				LOG_FATAL("file %s cannot be opened!",file_path_);

			unsigned int height = src.rows;
			unsigned int width = src.cols;
			unsigned int c_length = height * width;

			vec_t temp_(3*c_length);
			unsigned int index;
			for (unsigned int y = 0; y < height; y++)
				for (unsigned int x = 0; x < width; x++) {
					index = y * width + x;
					p_data[index] = ((mycnn::float_t) src.at<cv::Vec3b>(y, x)[0]);
					p_data[c_length + index] = ((mycnn::float_t) src.at<cv::Vec3b>(y, x)[1]);
					p_data[2*c_length + index] = ((mycnn::float_t) src.at<cv::Vec3b>(y, x)[2]);
				}

		}


#endif
		static void save_mean_file(mycnn::float_t *p_data, string mean_file_ , int length_)
		{
			ofstream os(mean_file_, ios::binary);
			os.precision(numeric_limits<mycnn::float_t>::digits10);
			if(!os)
				LOG_FATAL("file %s cannot be opened!",mean_file_.c_str());
			for(int i = 0 ; i < length_; ++i)
			{
				os.write((char*)(p_data+i), sizeof(mycnn::float_t));
			}
			os.close();
		}

		static void load_mean_file(mycnn::float_t *p_data, string mean_file_)
		{
			ifstream is(mean_file_);
			is.precision(numeric_limits<mycnn::float_t>::digits10);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",mean_file_.c_str());
			vector<mycnn::float_t> temp_;
			mycnn::float_t fp_;

			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(mycnn::float_t));
				temp_.push_back(fp_);
			}
			is.close();

			mycnn::float_t *d_= &temp_[0];
			memcpy(p_data,d_,temp_.size()*sizeof(mycnn::float_t));
		}

#if __PARALLELTYPE__ == __CUDA__
		static void load_mean_file_gpu(mycnn::float_t *p_data, string mean_file_)
		{
			ifstream is(mean_file_);
			is.precision(numeric_limits<mycnn::float_t>::digits10);
			if(!is)
				LOG_FATAL("file %s cannot be opened!",mean_file_.c_str());
			vector<mycnn::float_t> temp_;
			mycnn::float_t fp_;

			for(int i = 0 ;is.peek()!=EOF ;++i)
			{
				is.read(reinterpret_cast<char*>(&fp_), sizeof(mycnn::float_t));
				temp_.push_back(fp_);
			}
			is.close();

			mycnn::float_t *d_= &temp_[0];

			cuda_copy2dev(p_data,d_,temp_.size());
		}
#endif

	private:



	};
}
