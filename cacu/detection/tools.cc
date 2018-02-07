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

#include "tools.h"

using namespace std;
using namespace cacu;

namespace cacu_detection {

void NMS(vector<rect*> *&rects, float_t threshold, nms_type type) {

	if (rects->size() == 0)
		return;
	sort(rects->begin(), rects->end(), comp);
	vector<rect *> input_rects(*rects);
	rects->clear();

	rect *pRect = NULL;

	while (input_rects.size() > 0) {
		pRect = input_rects[0];
		rects->push_back(pRect);
		input_rects.erase(input_rects.begin());
		//printf("input_rects :%d ,rects :%d\n",input_rects.size(), rects->size());
		for (int i = 0; i < input_rects.size(); ++i) {
			switch (type) {
			case nms_iou:
				if (IOU(pRect, input_rects[i]) > threshold) {
					delete input_rects[i];
					input_rects.erase(input_rects.begin() + i);
					i--;
				}
				break;
			case nms_iom:
				if (IOM(pRect, input_rects[i]) > threshold) {
					delete input_rects[i];
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

}
