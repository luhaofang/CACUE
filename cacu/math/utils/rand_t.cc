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

#include "rand_t.h"

namespace cacu {

void set_rand_seed() {
	srand((unsigned int) time(NULL));
}

float_t gaussrand(float_t std, float_t m) {
	static float_t V1, V2, S;
	static int phase = 0;
	float_t X;
	if ( phase == 0 ) {
		do {
			float_t U1 = (float_t)rand() / RAND_MAX;
			float_t U2 = (float_t)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;
	return X * std + m;
}

float_t urand(float_t min, float_t max) {
	assert(min < max);
	float_t pRandomValue = ((float_t) rand() / (float_t) RAND_MAX);
	pRandomValue = pRandomValue * (max - min) + min;
	return pRandomValue;
}

float_t urand_clip(float_t min, float_t max, float_t abs_, float_t p_) {
	assert(p_ < 1);
	assert(p_ > 0);
	bool p = urand(0, 1) <= p_;
	float_t c = p ? urand(min, -abs_) : urand(abs_, max);
	return c;
}

float_t spherical_unrand(float_t std, float_t m, float_t min, float_t max){
	assert(min < max);
	float_t c = gaussrand(std, m);
	while(c <= min || c >= max){
		c = gaussrand(std, m);
	}
	return c;
}

unsigned int urandint(unsigned int min, unsigned int max) {

	assert(min < max);
	float_t pRandomValue = (float_t) (rand() % max) / max;
	pRandomValue = pRandomValue * (max - min) + min;
	return (unsigned int) pRandomValue;
}

unsigned int randint(int max) {
	assert(max >= 0);
	if(max == 0)
		return 0;
	return (unsigned int) (rand() % max);
}

}
