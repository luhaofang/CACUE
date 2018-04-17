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

#ifndef TOOLS_MTCNN_H_
#define TOOLS_MTCNN_H_

#include <math.h>

#include "../../cacu/detection/detection_definition.h"
#include "../../cacu/detection/rect.h"
#include "../../cacu/detection/tools.h"

#include "../../cacu/framework/blob.h"

using namespace cacu;
using namespace cacu_detection;

inline void rect2square(rect *rect_, dsize_t width, dsize_t height) {
	float_t w = (float_t) (rect_->r - rect_->l);
	float_t h = (float_t) (rect_->b - rect_->t);
	float_t l = std::max(w, h);
	rect_->l = std::max(0, (dsize_t) ((float_t) rect_->l + w * 0.5 - l * 0.5));
	rect_->t = std::max(0, (dsize_t) ((float_t) rect_->t + h * 0.5 - l * 0.5));
	rect_->r = (dsize_t) std::min((float_t) width, (float_t) rect_->l + l);
	rect_->b = (dsize_t) std::min((float_t) height, (float_t) rect_->t + l);
}

void generate_scales(dsize_t width, dsize_t height, vec_t *&scales) {
	float_t pr_scale = 1.0;
	dsize_t w, h;
	if (std::min(width, height) > 1000) {
		pr_scale = 1000.0 / std::min(width, height);
		w = dsize_t(width * pr_scale);
		h = dsize_t(height * pr_scale);
	} else if (std::max(width, height) < 1000) {
		pr_scale = 1000.0 / max(width, height);
		w = dsize_t(width * pr_scale);
		h = dsize_t(height * pr_scale);
	}

	float_t factor = 0.709;
	float_t factor_count = 0;
	dsize_t minl = std::min(h, w);
	while (minl >= 12) {
		scales->push_back(pr_scale * std::pow(factor, factor_count));
		minl *= factor;
		factor_count += 1;
	}
}

void detect_Pnet_face(blob *cls_prob, blob *roi, dsize_t max_side,
		float_t scale, dsize_t width, dsize_t height, float_t threshold,
		vector<rect*> *&proposal) {
	float_t block_size = 12.0;
	dsize_t in_side = 2 * max_side + 11;
	float_t stride = 0;
	dsize_t x, y;
	if (max_side != 1)
		stride = (float_t) (in_side - 12) / (max_side - 1);
#if __USE_DEVICE__ == ON
	float_t *pcls = cls_prob->s_data_cpu() + cls_prob->channel_length();
	float_t *proi = roi->s_data_cpu();
#else
	float_t *pcls = cls_prob->s_data() + cls_prob->channel_length();
	float_t *proi = roi->s_data();
#endif
	for (int i = 0; i < cls_prob->channel_length(); ++i) {
		if (pcls[i] >= threshold) {
			x = i % cls_prob->width();
			y = i / cls_prob->width();
			rect * boundingbox =
					new rect(
							(dsize_t) (stride * x * scale
									+ proi[i] * block_size * scale),
							(dsize_t) (stride * y * scale
									+ proi[roi->channel_length() + i]
											* block_size * scale),
							(dsize_t) ((stride * x + 11) * scale
									+ proi[roi->channel_length() * 2 + i]
											* block_size * scale),
							(dsize_t) ((stride * y + 11) * scale
									+ proi[roi->channel_length() * 3 + i]
											* block_size * scale), pcls[i]);
			rect2square(boundingbox, width, height);
			if (boundingbox->r > boundingbox->l
					&& boundingbox->b > boundingbox->t)
				proposal->push_back(boundingbox);
			else
				delete boundingbox;
		}
	}
}

void filter_Rnet_face(blob *cls_prob, blob *roi, dsize_t width, dsize_t height,
		float_t threshold, vector<rect*> *&proposal) {
#if __USE_DEVICE__ == ON
	float_t *pcls_s = cls_prob->s_data_cpu();
	float_t *proi_s = roi->s_data_cpu();
#else
	float_t *pcls_s = cls_prob->s_data();
	float_t *proi_s = roi->s_data();
#endif
	rect* rect_;
	float_t w, h;
	float_t *pcls, *proi;
	int index = 0;
	for (int i = 0; i < cls_prob->num(); ++i) {
		pcls = pcls_s + i * cls_prob->length();
		proi = proi_s + i * roi->length();
		rect_ = proposal->at(index);

		if (pcls[1] >= threshold) {
			w = (float_t) (rect_->r - rect_->l);
			h = (float_t) (rect_->b - rect_->t);
			rect_->l += (dsize_t) (proi[0] * w);
			rect_->t += (dsize_t) (proi[1] * h);
			rect_->r += (dsize_t) (proi[2] * w);
			rect_->b += (dsize_t) (proi[3] * h);
			rect_->score = pcls[1];
			rect2square(rect_, width, height);
			if (rect_->r <= rect_->l && rect_->b <= rect_->t) {
				delete rect_;
				proposal->erase(proposal->begin() + index);
				index--;
			}
		} else {
			delete rect_;
			proposal->erase(proposal->begin() + index);
			index--;
		}
		index++;
	}
	NMS(proposal, 0.7, nms_iou);
}

void filter_Onet_face(blob *cls_prob, blob *roi, dsize_t width, dsize_t height,
		float_t threshold, vector<rect*> *&proposal) {
#if __USE_DEVICE__ == ON
	float_t *pcls_s = cls_prob->s_data_cpu();
	float_t *proi_s = roi->s_data_cpu();
#else
	float_t *pcls_s = cls_prob->s_data();
	float_t *proi_s = roi->s_data();
#endif
	rect* rect_;
	float_t w, h;
	float_t *pcls, *proi;
	int index = 0;
	for (int i = 0; i < cls_prob->num(); ++i) {
		pcls = pcls_s + i * cls_prob->length();
		proi = proi_s + i * roi->length();
		rect_ = proposal->at(index);

		if (pcls[1] >= threshold) {
			w = (float_t) (rect_->r - rect_->l);
			h = (float_t) (rect_->b - rect_->t);
			rect_->l += (dsize_t) (proi[0] * w);
			rect_->t += (dsize_t) (proi[1] * h);
			rect_->r += (dsize_t) (proi[2] * w);
			rect_->b += (dsize_t) (proi[3] * h);
			rect_->score = pcls[1];
			rect2square(rect_, width, height);
			if (rect_->r <= rect_->l && rect_->b <= rect_->t) {
				delete rect_;
				proposal->erase(proposal->begin() + index);
				index--;
			}
		} else {
			delete rect_;
			proposal->erase(proposal->begin() + index);
			index--;
		}
		index++;
	}
	NMS(proposal, 0.7, nms_iom);
}


#endif
