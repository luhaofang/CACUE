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

#include "../cacu/math/math.h"

using namespace std;
using namespace cacu;

namespace cacu_tools {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
void imageio_utils::imread_gpu(cacu::float_t *p_data, const char* file_path_,
		const int size) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
	int c_length = height * width;

	vec_t temp_(3 * c_length);
	CHECK_EQ_OP(size, (int )temp_.size(),
			"image size %d must equal to the blob size %d!", (int )temp_.size(),
			size);
	int index;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			index = y * width + x;
			temp_[index] = ((cacu::float_t) src.at<cv::Vec3b>(y, x)[0]);
			temp_[c_length + index] =
					((cacu::float_t) src.at<cv::Vec3b>(y, x)[1]);
			temp_[2 * c_length + index] = ((cacu::float_t) src.at<cv::Vec3b>(y,
					x)[2]);
		}

	cuda_copy2dev(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
	src.release();
}

cacu::blob* imageio_utils::read_img2blob_gpu(const char* file_path_,
		phase_type phase_) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
	int c_length = height * width;

	vec_t temp_(3 * c_length);
	cacu::blob* img_blob = new cacu::blob(1, 3, width, height, 0, phase_);

	int index;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			index = y * width + x;
			temp_[index] = ((cacu::float_t) src.at<cv::Vec3b>(y, x)[0]);
			temp_[c_length + index] =
					((cacu::float_t) src.at<cv::Vec3b>(y, x)[1]);
			temp_[2 * c_length + index] = ((cacu::float_t) src.at<cv::Vec3b>(y,
					x)[2]);
		}

	cuda_copy2dev(img_blob->s_data(), &temp_[0], temp_.size());
	vec_t().swap(temp_);
	src.release();
	return img_blob;
}

void imageio_utils::resize_imread_gpu(cacu::float_t *p_data,
		const char* file_path_, int resize_h, int resize_w) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = resize_h;
	int width = resize_w;
	int c_length = height * width;
	cv::Mat dst;
	cv::resize(src, dst, cv::Size(resize_h, resize_w), cv::INTER_LINEAR);

	vec_t temp_(3 * c_length);
	int index;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			index = y * width + x;
			temp_[index] = ((cacu::float_t) dst.at<cv::Vec3b>(y, x)[0]);
			temp_[c_length + index] =
					((cacu::float_t) dst.at<cv::Vec3b>(y, x)[1]);
			temp_[2 * c_length + index] = ((cacu::float_t) dst.at<cv::Vec3b>(y,
					x)[2]);
		}

	cuda_copy2dev(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
	src.release();
	dst.release();
}

void imageio_utils::clip_imread_gpu(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w, bool flip,
		cacu::float_t *mean) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
	int _length = 3 * height * width;
	int channel_length = height * width;

	int c_length = clip_size_w * clip_size_h;
	cv::Mat dst;
	int flip_code = flip ? static_cast<int>(randint(2)) : 0;
	if (flip_code)
		cv::flip(src, dst, 1);
	else
		src.copyTo(dst);

	CHECK_LE_OP(clip_size_h, height,
			"clip height %d must less than the image height %d!", clip_size_h,
			height);
	CHECK_LE_OP(clip_size_w, width,
			"clip width %d must less than the image width %d!", clip_size_w,
			width);

	vec_t temp_(3 * c_length);
	int index;
	int start_w, start_h;
	start_w = randint(src.cols - clip_size_w);
	start_h = randint(src.rows - clip_size_h);
	cv::Vec3b setdata;

	if (mean != NULL) {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = dst.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0])
						- mean[(y + start_h) * width + x + start_w
								+ flip_code * (width - 1 - 2 * x + start_w)];
				temp_[c_length + index] = ((cacu::float_t) setdata[1])
						- mean[(y + start_h) * width + x + start_w
								+ flip_code * (width - 1 - 2 * x + start_w)
								+ channel_length];
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2])
						- mean[(y + start_h) * width + x + start_w
								+ flip_code * (width - 1 - 2 * x + start_w)
								+ 2 * channel_length];
			}
	} else {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = dst.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0]);
				temp_[c_length + index] = ((cacu::float_t) setdata[1]);
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2]);
			}
	}
	cuda_copy2dev(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
	src.release();
	dst.release();
}

void imageio_utils::center_clip_imread_gpu(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w,
		cacu::float_t *mean) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
	int _length = 3 * height * width;
	int channel_length = height * width;
	int c_length = clip_size_w * clip_size_h;

	CHECK_LE_OP(clip_size_h, height,
			"clip height %d must less than the image height %d!", clip_size_h,
			height);
	CHECK_LE_OP(clip_size_w, width,
			"clip width %d must less than the image width %d!", clip_size_w,
			width);

	vec_t temp_(3 * c_length);
	int index;
	int start_w, start_h;
	start_w = (src.cols - clip_size_w) / 2;
	start_h = (src.rows - clip_size_h) / 2;
	cv::Vec3b setdata;
	if (mean != NULL) {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = src.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0])
						- mean[(y + start_h) * width + x + start_w];
				temp_[c_length + index] = ((cacu::float_t) setdata[1])
						- mean[(y + start_h) * width + x + start_w
								+ channel_length];
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2])
						- mean[(y + start_h) * width + x + start_w
								+ 2 * channel_length];
			}
	} else {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = src.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0]);
				temp_[c_length + index] = ((cacu::float_t) setdata[1]);
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2]);
			}
	}
	cuda_copy2dev(p_data, &temp_[0], temp_.size());
	vec_t().swap(temp_);
	src.release();
}

void imageio_utils::imwrite_gpu(blob *blob_, const char* file_path_) {

	vec_t data_(blob_->count());
	device_copy2host(&data_[0], blob_->s_data(), blob_->count());
	int start_h, start_w, index;

	float_t max_s = data_[0], min_s = data_[0];

	for (int i = 1; i < data_.size(); ++i) {
		if (max_s < data_[i])
			max_s = data_[i];
		if (min_s > data_[i])
			min_s = data_[i];
	}

	float_t size_s = max_s - min_s;

	int h_size, w_size = sqrtf(blob_->num());
	h_size = blob_->num() / w_size;
	if (blob_->num() % w_size != 0 && w_size != 1)
		w_size += 1;
	cv::Mat src(cv::Size(blob_->width() * w_size, blob_->height() * h_size),
			CV_8UC3, cv::Scalar(0));

	for (int n = 0; n < blob_->num(); ++n) {
		start_h = n / w_size * blob_->height();
		start_w = n % w_size * blob_->width();
		for (int y = 0; y < blob_->height(); ++y)
			for (int x = 0; x < blob_->width(); ++x) {
				index = n * blob_->length() + y * blob_->width() + x;

				src.at<cv::Vec3b>(y + start_h, x + start_w)[0] = (data_[index]
						- min_s) / size_s * 255;
				src.at<cv::Vec3b>(y + start_h, x + start_w)[1] = (data_[index
						+ blob_->channel_length()] - min_s) / size_s * 255;
				src.at<cv::Vec3b>(y + start_h, x + start_w)[2] = (data_[index
						+ blob_->channel_length() * 2] - min_s) / size_s * 255;
			}
	}
	cv::imwrite(file_path_, src);
	LOG_INFO("Output image to : %s", file_path_);
	vec_t().swap(data_);
	src.release();
}

#endif
#endif

void imageio_utils::imread_cpu(cacu::float_t *p_data, const char* file_path_,
		const int p_size) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
	int c_length = height * width;

	vec_t temp_(3 * c_length);
	CHECK_EQ_OP(p_size, (int )temp_.size(),
			"image size %d must equal to the blob size %d!", (int )temp_.size(),
			p_size);
	int index;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			index = y * width + x;
			p_data[index] = ((cacu::float_t) src.at<cv::Vec3b>(y, x)[0]);
			p_data[c_length + index] =
					((cacu::float_t) src.at<cv::Vec3b>(y, x)[1]);
			p_data[2 * c_length + index] = ((cacu::float_t) src.at<cv::Vec3b>(y,
					x)[2]);
		}
	src.release();
}

cacu::blob* imageio_utils::read_img2blob_cpu(const char* file_path_,
		phase_type phase_) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
	int c_length = height * width;

	vec_t temp_(3 * c_length);

	cacu::blob* img_blob = new cacu::blob(1, 3, width, height, 0, phase_);

	float_t *p_data = img_blob->s_data();
	int index;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			index = y * width + x;
			p_data[index] = ((cacu::float_t) src.at<cv::Vec3b>(y, x)[0]);
			p_data[c_length + index] =
					((cacu::float_t) src.at<cv::Vec3b>(y, x)[1]);
			p_data[2 * c_length + index] = ((cacu::float_t) src.at<cv::Vec3b>(y,
					x)[2]);
		}
	src.release();
	return img_blob;
}

void imageio_utils::resize_imread_cpu(cacu::float_t *p_data,
		const char* file_path_, int resize_h, int resize_w) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = resize_h;
	int width = resize_w;
	int c_length = height * width;

	cv::Mat dst;
	cv::resize(src, dst, cv::Size(resize_h, resize_w), cv::INTER_LINEAR);

	vec_t temp_(3 * c_length);
	int index;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			index = y * width + x;
			p_data[index] = ((cacu::float_t) dst.at<cv::Vec3b>(y, x)[0]);
			p_data[c_length + index] =
					((cacu::float_t) dst.at<cv::Vec3b>(y, x)[1]);
			p_data[2 * c_length + index] = ((cacu::float_t) dst.at<cv::Vec3b>(y,
					x)[2]);
		}
	src.release();
	dst.release();
}

void imageio_utils::clip_imread_cpu(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w, bool flip,
		cacu::float_t *mean) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
//		int _length = 3 * height * width;
	int channel_length = height * width;

	int c_length = clip_size_w * clip_size_h;
	cv::Mat dst;
	int flip_code = flip ? static_cast<int>(randint(2)) : 0;
	if (flip_code)
		cv::flip(src, dst, 1);
	else
		src.copyTo(dst);

	CHECK_LE_OP(clip_size_h, height,
			"clip height %d must less than the image height %d!", clip_size_h,
			height);
	CHECK_LE_OP(clip_size_w, width,
			"clip width %d must less than the image width %d!", clip_size_w,
			width);

	vec_t temp_(3 * c_length);
	int index;
	int start_w, start_h;
	start_w = randint(src.cols - clip_size_w);
	start_h = randint(src.rows - clip_size_h);
	cv::Vec3b setdata;
	if (mean != NULL) {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = dst.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0])
						- mean[(y + start_h) * width + x + start_w
								+ flip_code * (width - 1 - 2 * x + start_w)];
				temp_[c_length + index] = ((cacu::float_t) setdata[1])
						- mean[(y + start_h) * width + x + start_w
								+ flip_code * (width - 1 - 2 * x + start_w)
								+ channel_length];
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2])
						- mean[(y + start_h) * width + x + start_w
								+ flip_code * (width - 1 - 2 * x + start_w)
								+ 2 * channel_length];
			}
	} else {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = dst.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0]);
				temp_[c_length + index] = ((cacu::float_t) setdata[1]);
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2]);
			}
	}
	memcpy(p_data, &temp_[0], temp_.size() * sizeof(float_t));
	vec_t().swap(temp_);
	src.release();
	dst.release();
}

void imageio_utils::center_clip_imread_cpu(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w,
		cacu::float_t *mean) {
	cv::Mat src = cv::imread(file_path_, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_path_);

	int height = src.rows;
	int width = src.cols;
//		int _length = 3 * height * width;
	int channel_length = height * width;
	int c_length = clip_size_w * clip_size_h;

	CHECK_LE_OP(clip_size_h, height,
			"clip height %d must less than the image height %d!", clip_size_h,
			height);
	CHECK_LE_OP(clip_size_w, width,
			"clip width %d must less than the image width %d!", clip_size_w,
			width);

	vec_t temp_(3 * c_length);
	int index;
	int start_w, start_h;
	start_w = (src.cols - clip_size_w) / 2;
	start_h = (src.rows - clip_size_h) / 2;
	cv::Vec3b setdata;
	if (mean != NULL) {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = src.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0])
						- mean[(y + start_h) * width + x + start_w];
				temp_[c_length + index] = ((cacu::float_t) setdata[1])
						- mean[(y + start_h) * width + x + start_w
								+ channel_length];
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2])
						- mean[(y + start_h) * width + x + start_w
								+ 2 * channel_length];
			}
	} else {
		for (int y = 0; y < clip_size_h; y++)
			for (int x = 0; x < clip_size_w; x++) {
				index = y * clip_size_w + x;
				setdata = src.at<cv::Vec3b>(y + start_h, x + start_w);
				temp_[index] = ((cacu::float_t) setdata[0]);
				temp_[c_length + index] = ((cacu::float_t) setdata[1]);
				temp_[2 * c_length + index] = ((cacu::float_t) setdata[2]);
			}
	}
	memcpy(p_data, &temp_[0], temp_.size() * sizeof(float_t));
	vec_t().swap(temp_);
	src.release();
}

void imageio_utils::imwrite_cpu(blob *blob_, const char* file_path_) {
	cacu::float_t max_s = blob_->s_data()[0], min_s = blob_->s_data()[0];

	for (int i = 1; i < blob_->count(); ++i) {
		if (max_s < blob_->s_data()[i])
			max_s = blob_->s_data()[i];
		if (min_s > blob_->s_data()[i])
			min_s = blob_->s_data()[i];
	}

	cacu::float_t size_s = max_s - min_s;

	int h_size, w_size = sqrtf(blob_->num());
	h_size = blob_->num() / w_size;
	if (blob_->num() % w_size != 0 && w_size != 1)
		w_size += 1;
	cv::Mat src(cv::Size(blob_->width() * w_size, blob_->height() * h_size),
			CV_8UC3, cv::Scalar(0));
	int start_h, start_w, index;
	//cv::Vec3b setdata;
	for (int n = 0; n < blob_->num(); n++) {
		start_h = n / w_size * blob_->height();
		start_w = n % w_size * blob_->width();
		for (int y = 0; y < blob_->height(); y++)
			for (int x = 0; x < blob_->width(); x++) {
				index = n * blob_->length() + y * blob_->width() + x;
				//setdata = ;
				src.at<cv::Vec3b>(y + start_h, x + start_w)[0] =
						(blob_->s_data()[index] - min_s) / size_s * 255;
				src.at<cv::Vec3b>(y + start_h, x + start_w)[1] =
						(blob_->s_data()[index + blob_->channel_length()]
								- min_s) / size_s * 255;
				src.at<cv::Vec3b>(y + start_h, x + start_w)[2] =
						(blob_->s_data()[index + blob_->channel_length() * 2]
								- min_s) / size_s * 255;
			}
	}
	cv::imwrite(file_path_, src);
	LOG_INFO("Output image to : %s", file_path_);
	src.release();
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
		const string& mean_file_, int length_) {

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
		const string& mean_file_) {
	ifstream is(mean_file_, ios::binary);
	is.precision(numeric_limits<cacu::float_t>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", mean_file_.c_str());

	for (int i = 0; is.peek() != EOF; ++i) {
		is.read(reinterpret_cast<char*>(p_data + i), sizeof(cacu::float_t));
	}
	is.close();
}

void imageio_utils::imwrite(blob *blob_, const char* file_path_) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	imwrite_gpu(blob_, file_path_);
#endif
#else
	imwrite_cpu(blob_, file_path_);
#endif
}

void imageio_utils::resize_imread(cacu::float_t *p_data, const char* file_path_,
		int resize_h, int resize_w) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	resize_imread_gpu(p_data, file_path_, resize_h, resize_w);
#endif
#else
	resize_imread_cpu(p_data, file_path_, resize_h, resize_w);
#endif
}

void imageio_utils::imread(cacu::float_t *p_data, const char* file_path_,
		const int p_size) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	imread_gpu(p_data, file_path_, p_size);
#endif
#else
	imread_cpu(p_data, file_path_, p_size);
#endif
}

void imageio_utils::clip_imread(cacu::float_t *p_data, const char* file_path_,
		int clip_size_h, int clip_size_w, bool flip) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	clip_imread_gpu(p_data, file_path_, clip_size_h, clip_size_w, flip);
#endif
#else
	clip_imread_cpu(p_data, file_path_, clip_size_h, clip_size_w, flip);
#endif
}

void imageio_utils::center_clip_imread(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	center_clip_imread_gpu(p_data, file_path_, clip_size_h, clip_size_w);
#endif
#else
	center_clip_imread_cpu(p_data, file_path_, clip_size_h, clip_size_w);
#endif
}

void imageio_utils::clip_imread_channel_wise_norm(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w,
		const vec_t &channels, bool flip) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	clip_imread_gpu(p_data, file_path_, clip_size_h, clip_size_w, flip);
#endif
#else
	clip_imread_cpu(p_data, file_path_, clip_size_h, clip_size_w, flip);
#endif
	for (int c = 0; c < 3; ++c)
		cacu_sdxsize(p_data + c * clip_size_h * clip_size_w,
				clip_size_h * clip_size_w, -channels[c], (cacu::float_t) 1,
				p_data + c * clip_size_h * clip_size_w);
}

void imageio_utils::center_clip_imread_channel_wise_norm(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w,
		const vec_t &channels) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	center_clip_imread_gpu(p_data, file_path_, clip_size_h, clip_size_w);
#endif
#else
	center_clip_imread_cpu(p_data, file_path_, clip_size_h, clip_size_w);
#endif
	for (int c = 0; c < 3; ++c)
		cacu_sdxsize(p_data + c * clip_size_h * clip_size_w,
				clip_size_h * clip_size_w, -channels[c], (cacu::float_t) 1,
				p_data + c * clip_size_h * clip_size_w);
}

void imageio_utils::imread_channel_wise_norm(cacu::float_t *p_data,
		const char* file_path_, int p_size, const vec_t &channels) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	imread_gpu(p_data, file_path_, p_size);
#endif
#else
	imread_cpu(p_data, file_path_, p_size);
#endif
	for (int c = 0; c < 3; ++c)
		cacu_sdxsize(p_data + c * p_size / 3, p_size / 3, -channels[c],
				(cacu::float_t) 1, p_data + c * p_size / 3);
}

void imageio_utils::clip_imread_element_wise_norm(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w,
		cacu::float_t *mean, bool flip) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	clip_imread_gpu(p_data, file_path_, clip_size_h, clip_size_w, flip, mean);
#endif
#else
	clip_imread_cpu(p_data, file_path_, clip_size_h, clip_size_w, flip, mean);
#endif
}

void imageio_utils::center_clip_imread_element_wise_norm(cacu::float_t *p_data,
		const char* file_path_, int clip_size_h, int clip_size_w,
		cacu::float_t *mean) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	center_clip_imread_gpu(p_data, file_path_, clip_size_h, clip_size_w, mean);
#endif
#else
	center_clip_imread_cpu(p_data, file_path_, clip_size_h, clip_size_w, mean);
#endif
}

cacu::blob* imageio_utils::read_img2blob(const char *file_path_,
		phase_type phase_) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	return read_img2blob_gpu(file_path_, phase_);
#endif
#else
	return read_img2blob_cpu(file_path_, phase_);
#endif
}

}
