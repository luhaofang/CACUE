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


#ifndef CACU_MATH_ARMV7A_TEST_HPP_
#define CACU_MATH_ARMV7A_TEST_HPP_

#include <arm_neon.h>

uint32x4_t double_elements(uint32x4_t input)
{
	 return(vaddq_u32(input,input));
}

void cross_product_s(float32_t *r, float32_t* a, float32_t* b) {

	float32x2_t vec_a_1 = vld1_f32(a + 1);
	float32x2_t vec_a_2 = vld1_f32(a);
	float32x2_t vec_b_1 = vld1_f32(b + 1);
	float32x2_t vec_b_2 = vld1_f32(b);

	float32x4_t vec_a = vcombine_f32(vec_a_1, vec_a_2);
	float32x4_t vec_b = vcombine_f32(vec_b_1, vec_b_2);
	float32x4_t vec_a_rot = vextq_f32(vec_a, vec_a, 1);
	float32x4_t vec_b_rot = vextq_f32(vec_b, vec_b, 1);

	float32x4_t prod = vmulq_f32(vec_a, vec_b_rot);
	prod = vmlsq_f32(prod, vec_a_rot, vec_b);

	vst1_f32(r, vget_low_f32(prod)); // Store the lower two elements to address r
	vst1_lane_f32(r + 2, vget_high_f32(prod), 0);
}

void cross_product_q(float32_t* r, float32_t* a, float32_t* b) {

    float32x4x3_t vec_a = vld3q_f32(a);
    float32x4x3_t vec_b = vld3q_f32(b);
    float32x4x3_t result;
    result.val[0] = vmulq_f32(vec_a.val[1], vec_b.val[2]);
    result.val[0] = vmlsq_f32(result.val[0], vec_a.val[2], vec_b.val[1]);
    result.val[1] = vmulq_f32(vec_a.val[2], vec_b.val[0]);
    result.val[1] = vmlsq_f32(result.val[1], vec_a.val[0], vec_b.val[2]);
    result.val[2] = vmulq_f32(vec_a.val[0], vec_b.val[1]);
    result.val[2] = vmlsq_f32(result.val[2], vec_a.val[1], vec_b.val[0]);
    vst3q_f32(r, result);
}


void cross_product(float32_t* r, float32_t* a, float32_t* b, int count) {
	int count4 = count / 4;
	count &= 3;
	while(count4 > 0) {
		cross_product_q(r, a, b);
		r += 12;
		a += 12;
		b += 12;
		count4--;
	}
	if(count == 1) {
		cross_product_s(r, a, b);
	}
}


#endif /* CACU_MATH_ARMV7A_TEST_HPP_ */
