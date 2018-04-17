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

#include "imageio_utils.h"

using namespace std;
using namespace cacu;

namespace cacu_tools {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
void imageio_utils::imread_gpu(cacu::float_t *p_data,
		const char* file_path_, const int size) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	unsigned int height = src.rows;
	unsigned int width = src.cols;
	unsigned int c_length = height * width;

	vec_t temp_(3 * c_length);
	CHECK_EQ_OP(size, temp_.size(),
			"image size %d must equal to the blob size %d!", temp_.size(),
			size);
	unsigned int index;
	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			index = y * width + x;
			temp_[index] = ((cacu::float_t) src.at<cv::Vec3b>(y, x)[0]);
			temp_[c_length + index] =
					((cacu::float_t) src.at<cv::Vec3b>(y, x)[1]);
			temp_[2 * c_length + index] = ((cacu::float_t) src.at<cv::Vec3b>(y,
					x)[2]);
		}

	cuda_copy2dev(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
}

void imageio_utils::resize_imread_gpu(cacu::float_t *p_data,
		const char* file_path_, int resize_h, int resize_w) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	unsigned int height = resize_h;
	unsigned int width = resize_w;
	unsigned int c_length = height * width;
	cv::Mat dst;
	cv::resize(src, dst, cv::Size(resize_w, resize_h), (0, 0), (0, 0),
			cv::INTER_LINEAR);

	vec_t temp_(3 * c_length);
	unsigned int index;
	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			index = y * width + x;
			temp_[index] = ((cacu::float_t) dst.at<cv::Vec3b>(y, x)[0]);
			temp_[c_length + index] =
					((cacu::float_t) dst.at<cv::Vec3b>(y, x)[1]);
			temp_[2 * c_length + index] = ((cacu::float_t) dst.at<cv::Vec3b>(y,
					x)[2]);
		}

	cuda_copy2dev(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
}

void imageio_utils::clip_imread_gpu(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	unsigned int height = src.rows;
	unsigned int width = src.cols;
	unsigned int c_length = clip_size_w * clip_size_h;
	cv::Mat dst = src;

	CHECK_LT_OP(clip_size_h, height,
			"clip height %d must less than the image height %d!", clip_size_h,
			height);
	CHECK_LT_OP(clip_size_w, width,
			"clip width %d must less than the image width %d!", clip_size_w,
			width);

	vec_t temp_(3 * c_length);
	unsigned int index;
	unsigned int start_w, start_h;
	start_w = randint(src.cols - clip_size_w);
	start_h = randint(src.rows - clip_size_h);
	cv::Vec3b setdata;
	for (unsigned int y = 0; y < clip_size_h; y++)
		for (unsigned int x = 0; x < clip_size_w; x++) {
			index = y * clip_size_w + x;
			setdata = dst.at<cv::Vec3b>(y + start_h, x + start_w);
			temp_[index] = ((cacu::float_t) setdata[0]);
			temp_[c_length + index] = ((cacu::float_t) setdata[1]);
			temp_[2 * c_length + index] = ((cacu::float_t) setdata[2]);
		}
	cuda_copy2dev(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
}
#endif
#endif

void imageio_utils::imread(cacu::float_t *p_data, const char* file_path_,
		const int p_size) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	unsigned int height = src.rows;
	unsigned int width = src.cols;
	unsigned int c_length = height * width;

	vec_t temp_(3 * c_length);
	CHECK_EQ_OP(p_size, temp_.size(),
			"image size %d must equal to the blob size %d!", temp_.size(),
			p_size);
	unsigned int index;
	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			index = y * width + x;
			p_data[index] = ((cacu::float_t) src.at<cv::Vec3b>(y, x)[0]);
			p_data[c_length + index] =
					((cacu::float_t) src.at<cv::Vec3b>(y, x)[1]);
			p_data[2 * c_length + index] = ((cacu::float_t) src.at<cv::Vec3b>(y,
					x)[2]);
		}
}

void imageio_utils::resize_imread(cacu::float_t *p_data,
		const char* file_path_, int resize_h, int resize_w) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	unsigned int height = resize_h;
	unsigned int width = resize_w;
	unsigned int c_length = height * width;

	cv::Mat dst;
	cv::resize(src, dst, cv::Size(resize_w, resize_h), (0, 0), (0, 0),
			cv::INTER_LINEAR);

	vec_t temp_(3 * c_length);
	unsigned int index;
	for (unsigned int y = 0; y < height; y++)
		for (unsigned int x = 0; x < width; x++) {
			index = y * width + x;
			p_data[index] = ((cacu::float_t) dst.at<cv::Vec3b>(y, x)[0]);
			p_data[c_length + index] =
					((cacu::float_t) dst.at<cv::Vec3b>(y, x)[1]);
			p_data[2 * c_length + index] = ((cacu::float_t) dst.at<cv::Vec3b>(y,
					x)[2]);
		}
}

void imageio_utils::clip_imread(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	unsigned int height = src.rows;
	unsigned int width = src.cols;
	unsigned int c_length = clip_size_w * clip_size_h;
	cv::Mat dst = src;

	CHECK_LT_OP(clip_size_h, height,
			"clip height %d must less than the image height %d!", clip_size_h,
			height);
	CHECK_LT_OP(clip_size_w, width,
			"clip width %d must less than the image width %d!", clip_size_w,
			width);

	vec_t temp_(3 * c_length);
	unsigned int index;
	unsigned int start_w, start_h;
	start_w = randint(src.cols - clip_size_w);
	start_h = randint(src.rows - clip_size_h);
	cv::Vec3b setdata;
	for (unsigned int y = 0; y < clip_size_h; y++)
		for (unsigned int x = 0; x < clip_size_w; x++) {
			index = y * clip_size_w + x;
			setdata = dst.at<cv::Vec3b>(y + start_h, x + start_w);
			temp_[index] = ((cacu::float_t) setdata[0]);
			temp_[c_length + index] = ((cacu::float_t) setdata[1]);
			temp_[2 * c_length + index] = ((cacu::float_t) setdata[2]);
		}
	memcpy(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
}
//*/

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
void imageio_utils::load_mean_file_gpu(cacu::float_t *p_data,
		string mean_file_) {
	ifstream is(mean_file_);
	is.precision(numeric_limits<cacu::float_t>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", mean_file_.c_str());
	vector<cacu::float_t> temp_;
	cacu::float_t fp_;

	for (int i = 0; is.peek() != EOF; ++i) {
		is.read(reinterpret_cast<char*>(&fp_), sizeof(cacu::float_t));
		temp_.push_back(fp_);
	}
	is.close();

	cacu::float_t *d_ = &temp_[0];

	cuda_copy2dev(p_data, d_, temp_.size());
}
#endif
#endif
void imageio_utils::save_mean_file(cacu::float_t *p_data,
		string mean_file_, int length_) {
	ofstream os(mean_file_, ios::binary);
	os.precision(numeric_limits<cacu::float_t>::digits10);
	if (!os)
		LOG_FATAL("file %s cannot be opened!", mean_file_.c_str());
	for (int i = 0; i < length_; ++i) {
		os.write((char*) (p_data + i), sizeof(cacu::float_t));
	}
	os.close();
}

void imageio_utils::load_mean_file(cacu::float_t *p_data,
		string mean_file_) {
	ifstream is(mean_file_, ios::binary);
	is.precision(numeric_limits<cacu::float_t>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", mean_file_.c_str());

	for (int i = 0; is.peek() != EOF; ++i) {
		is.read(reinterpret_cast<char*>(p_data + i), sizeof(cacu::float_t));
	}
	is.close();
}
}
