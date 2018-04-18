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

#ifndef DEMO_HPP_
#define DEMO_HPP_

#include <libpng/png.h>
#include <opencv2/opencv.hpp>

#include <string>

//#include "../../cacu/definition.h"
//#include "../../cacu/utils/log.h"

#include "tools_mtcnn.h"
#include "models_creater.h"

using namespace cv;
using namespace std;
using namespace cacu;

//#ifdef linux

void face_detect(chars_t file_name) {

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(1);
#endif
#endif
	set_rand_seed();
	network *Pnet = create_Pnet(1, test);
	//Pnet->load_weights(
	//		"/home/seal/cuda-workspace/CACUE_N/example/mtcnn/model/Pnet.model");
	Pnet->load_weights("/home/haofang/experiment/mtcnn/models/Pnet.model");
	network *Rnet = create_Rnet(1, test);
	//Rnet->load_weights(
	//		"/home/seal/cuda-workspace/CACUE_N/example/mtcnn/model/Rnet.model");
	Rnet->load_weights("/home/haofang/experiment/mtcnn/models/Rnet.model");
	network *Onet = create_Onet(1, test);
	//Onet->load_weights(
	//		"/home/seal/cuda-workspace/CACUE_N/example/mtcnn/model/Onet.model");
	Onet->load_weights("/home/haofang/experiment/mtcnn/models/Onet.model");

	float_t thresholds[3] = { 0.5, 0.5, 0.6 };
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
	cv::Mat src = cv::imread(file_name, cv::IMREAD_COLOR);
	if (!src.data)
		LOG_FATAL("file %s cannot be opened!", file_name.c_str());
	cv::Mat dst;
	vec_t temp_(1);
	dsize_t height = src.rows;
	dsize_t width = src.cols;

	vec_t *scales = new vec_t();
	generate_scales(width, height, scales);

	vector<rect*> *proposals = new vector<rect*>();
	//Pnet processing
	blob* cls_prob = NULL;
	blob* roi = NULL;
	dsize_t index, hs, ws;
	for (dsize_t i = 0; i < scales->size(); ++i) {
		hs = dsize_t(height * scales->at(i));
		ws = dsize_t(width * scales->at(i));
		cv::resize(src, dst, cv::Size(ws, hs), (0, 0), (0, 0),
				cv::INTER_LINEAR);
		Pnet->input_blobs()->at(0)->resize(1, 3, ws, hs);
		temp_.resize(3 * ws * hs);
		for (dsize_t y = 0; y < hs; y++)
			for (dsize_t x = 0; x < ws; x++) {
				index = y * ws + x;
				temp_[index] = (((float_t) dst.at<cv::Vec3b>(y, x)[0] - 127.5)
						/ 128);
				temp_[ws * hs + index] = (((float_t) dst.at<cv::Vec3b>(y, x)[1]
						- 127.5) / 128);
				temp_[2 * ws * hs + index] =
						(((float_t) dst.at<cv::Vec3b>(y, x)[2] - 127.5) / 128);
			}
		((blob *) Pnet->input_blobs()->at(0))->copy2data(temp_);
		vec_t().swap(temp_);
		Pnet->forward_propagate();
		cls_prob = Pnet->output_blob();
		roi = Pnet->get_op(Pnet->op_count() - 2)->out_data<blob>();
		detect_Pnet_face(cls_prob, roi,
				std::max(cls_prob->width(), cls_prob->height()),
				1.0 / scales->at(i), width, height, thresholds[0], proposals);
	}

	NMS(proposals, 0.7, nms_iou);
	LOG_DEBUG("proposal: %d", proposals->size());

	rect *rect_;
	cv::Mat rectangle;

	//Rnet processing
	if (proposals->size() != 0) {
		Rnet->input_blobs()->at(0)->resize(proposals->size(), 3, 24, 24);

		for (dsize_t i = 0; i < proposals->size(); ++i) {
			rect_ = proposals->at(i);
			src(
					cv::Rect(rect_->l, rect_->t, (rect_->r - rect_->l),
							(rect_->b - rect_->t))).copyTo(rectangle);
			cv::resize(rectangle, dst, cv::Size(24, 24), (0, 0), (0, 0),
					cv::INTER_LINEAR);
			temp_.resize(3 * 24 * 24);
			for (dsize_t y = 0; y < 24; y++)
				for (dsize_t x = 0; x < 24; x++) {
					index = y * 24 + x;
					temp_[index] = (((float_t) dst.at<cv::Vec3b>(y, x)[0]
							- 127.5) / 128);
					temp_[24 * 24 + index] =
							(((float_t) dst.at<cv::Vec3b>(y, x)[1] - 127.5)
									/ 128);
					temp_[2 * 24 * 24 + index] = (((float_t) dst.at<cv::Vec3b>(
							y, x)[2] - 127.5) / 128);
				}
			((blob *) Rnet->input_blobs()->at(0))->copy2data(temp_, i);
			vec_t().swap(temp_);
		}
		Rnet->forward_propagate();
		cls_prob = Rnet->output_blob();
		roi = Rnet->get_op(Rnet->op_count() - 2)->out_data<blob>();
		filter_Rnet_face(cls_prob, roi, width, height, thresholds[1],
				proposals);
	}
	LOG_DEBUG("proposal: %d", proposals->size());
	//Onet processing
	if (proposals->size() != 0) {

		Onet->input_blobs()->at(0)->resize(proposals->size(), 3, 48, 48);
		for (dsize_t i = 0; i < proposals->size(); ++i) {
			rect_ = proposals->at(i);
			src(
					cv::Rect(rect_->l, rect_->t, (rect_->r - rect_->l),
							(rect_->b - rect_->t))).copyTo(rectangle);
			cv::resize(rectangle, dst, cv::Size(48, 48), (0, 0), (0, 0),
					cv::INTER_LINEAR);
			temp_.resize(3 * 48 * 48);
			for (dsize_t y = 0; y < 48; y++)
				for (dsize_t x = 0; x < 48; x++) {
					index = y * 48 + x;
					temp_[index] = (((float_t) dst.at<cv::Vec3b>(y, x)[0]
							- 127.5) / 128);
					temp_[48 * 48 + index] =
							(((float_t) dst.at<cv::Vec3b>(y, x)[1] - 127.5)
									/ 128);
					temp_[2 * 48 * 48 + index] = (((float_t) dst.at<cv::Vec3b>(
							y, x)[2] - 127.5) / 128);
				}
			((blob *) Onet->input_blobs()->at(0))->copy2data(temp_, i);
			vec_t().swap(temp_);
		}

		Onet->forward_propagate();
		cls_prob = Onet->output_blob();
		roi = Onet->get_op(Onet->op_count() - 2)->out_data<blob>();
		filter_Onet_face(cls_prob, roi, width, height, thresholds[2],
				proposals);
	}
	LOG_DEBUG("proposal: %d", proposals->size());
	gettimeofday(&end, NULL);
	unsigned long diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
		- start.tv_usec;
	LOG_INFO("time cost:%ld ms", diff / 1000);
	for (dsize_t i = 0; i < proposals->size(); ++i) {
		cv::rectangle(src,
				cv::Rect(proposals->at(i)->l, proposals->at(i)->t,
						(proposals->at(i)->r - proposals->at(i)->l),
						(proposals->at(i)->b - proposals->at(i)->t)),
				cv::Scalar(0, 255, 0), 1, 1, 0);
	}
	cv::imshow("dst", src);
	cv::waitKey();

	delete Pnet;
	delete Rnet;
	delete Onet;
	delete scales;
	delete proposals;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}

//#endif

#endif
