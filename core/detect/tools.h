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

#include <vector>

using namespace std;

namespace mycnn_detection{


	enum nms_type{
		nms_iou,
		nms_iom
	};

	inline bool comp(const rect *a, const rect *b){
			return a->score >= b->score;
	}

	inline float_t IOU(rect* rect1, rect* rect2)
	{
		size_t intersection = max(0, min(rect1->r,rect2->r) - max(rect1->l,rect2->l)) * max(0, min(rect1->b,rect2->b) - max(rect1->t,rect2->t));
		size_t _union = (rect1->r - rect1->l) * (rect1->b - rect1->t) + (rect2->r - rect2->l) * (rect2->b - rect2->t) - intersection;
		if(_union == 0)
			return 0;
		else
			return (float_t)(intersection)/_union;
	}

	inline float_t IOM(rect* rect1, rect* rect2)
	{
		size_t intersection = max(0, min(rect1->r,rect2->r) - max(rect1->l,rect2->l)) * max(0, min(rect1->b,rect2->b) - max(rect1->t,rect2->t));
		size_t min_area = min((rect1->r - rect1->l) * (rect1->b - rect1->t),(rect2->r - rect2->l) * (rect2->b - rect2->t));
		return (float_t)(intersection)/min_area;
	}

	inline void NMS(vector<rect *> &rects, float_t threshold, nms_type type)
	{

		if(rects.size() == 0)
			return;
		sort(rects.begin(),rects.end(),comp);
		vector<size_t> areas(rects.size());
		vector<rect *> input_rects(rects);
		rects.clear();

		rect *pRect = NULL;

		while(input_rects.size() > 0){
			pRect = input_rects[0];
			//LOG_DEBUG("%d,%d,%d,%d,%f", pRect->l,pRect->t,pRect->r,pRect->b,pRect->score);
			rects.push_back(pRect);
			input_rects.erase(input_rects.begin());
			for(int i = 0; i < input_rects.size(); ++i)
			{
				switch(type)
				{
				case nms_iou:
					if(IOU(pRect,input_rects[i])>=threshold){
						input_rects.erase(input_rects.begin() + i);
						i--;
					}
					break;
				case nms_iom:
					if(IOM(pRect,input_rects[i])>=threshold){
						input_rects.erase(input_rects.begin() + i);
						i--;
					}
					break;
				default:
					break;
				}
			}
		}
	}



};