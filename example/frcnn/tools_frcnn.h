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

#ifndef TOOLS_FRCNN_H_
#define TOOLS_FRCNN_H_

#include "../../cacu/detection/detection_definition.h"
#include "../../cacu/detection/rect.h"
#include "../../cacu/detection/tools.h"

#include "../../cacu/framework/blob.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cacu;
using namespace cacu_detection;

void readimg(blob *&blob_, string file_name) {

	cv::Mat src = cv::imread(file_name, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_name.c_str());
	vec_t temp_(src.rows * src.cols * 3);

	int index;

	for (dsize_t y = 0; y < src.rows; y++)
		for (dsize_t x = 0; x < src.cols; x++) {
			index = y * src.cols + x;
			temp_[index] =
					((cacu::float_t) src.at < cv::Vec3b > (y, x)[0] - 127.5) / 127.5;
			temp_[src.cols * src.rows + index] = ((cacu::float_t) src.at < cv::Vec3b
					> (y, x)[1] - 127.5) / 127.5;
			temp_[2 * src.cols * src.rows + index] = ((cacu::float_t) src.at
					< cv::Vec3b > (y, x)[2] - 127.5) / 127.5;
		}
	blob_->resize(1, 3, src.cols, src.rows);
	blob_->copy2data(temp_);
}

void resize_readimg(cacu::float_t *_pdata, string file_name, int resize_w,int resize_h) {

	cv::Mat src = cv::imread(file_name, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_name.c_str());

	unsigned int height = resize_h;
	unsigned int width = resize_w;
	unsigned int c_length = height * width;
	cv::Mat dst;
	cv::resize(src, dst, cv::Size(resize_w, resize_h), cv::INTER_LINEAR);

	vec_t temp_(3 * c_length);
	unsigned int index;
	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			index = y * width + x;
			temp_[index] = ((cacu::float_t) dst.at<cv::Vec3b>(y, x)[0] - 127.5) / 127.5;
			temp_[c_length + index] =
					((cacu::float_t) dst.at<cv::Vec3b>(y, x)[1] - 127.5) / 127.5;
			temp_[2 * c_length + index] = ((cacu::float_t) dst.at<cv::Vec3b>(y,
					x)[2] - 127.5) / 127.5;
		}
#if __USE_DEVICE__ == ON
	device_copy2dev(_pdata, &temp_[0], temp_.size());
#else
	cacu_copy_cpu(&temp_[0],temp_.size(), _pdata);
#endif
}

void readimg(cacu::float_t *_pdata, string file_name) {

	cv::Mat src = cv::imread(file_name, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_name.c_str());

	unsigned int height = src.rows;
	unsigned int width = src.cols;
	unsigned int c_length = height * width;
	cv::Mat dst = src;

	vec_t temp_(3 * c_length);
	unsigned int index;
	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			index = y * width + x;
			temp_[index] = ((cacu::float_t) dst.at<cv::Vec3b>(y, x)[0] - 127.5) / 127.5;
			temp_[c_length + index] =
					((cacu::float_t) dst.at<cv::Vec3b>(y, x)[1] - 127.5) / 127.5;
			temp_[2 * c_length + index] = ((cacu::float_t) dst.at<cv::Vec3b>(y,
					x)[2] - 127.5) / 127.5;
		}
#if __USE_DEVICE__ == ON
	device_copy2dev(_pdata, &temp_[0], temp_.size());
#else
	cacu_copy_cpu(&temp_[0],temp_.size(), _pdata);
#endif
}


#endif
