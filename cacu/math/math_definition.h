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

#ifndef MATH_DEFINITION_H_
#define MATH_DEFINITION_H_

namespace cacu{

#define PI 3.1415926

typedef enum {

	TRANS, NOTRANS

} TRANSPOSE;

typedef enum {

	BYWIDTH, BYHEIGHT

} SUM;

/*
 * regularization type for weights update in optimization methods.
 *
 */
enum regularize_type {
	L1,
	L2,
	NO_REGULARIZATION
};

/*
 * regularization type for weights update in optimization methods.
 *
 */
enum norm_type {
	norm_l1,
	norm_l2
};

enum grad_direction_type {
	minimize,
	maximize
};

enum rotate_code {
	rotate_top,
	rotate_left,
	rotate_bottom,
	rotate_right
};

inline int Factorial(int m){
	if(m == 0)
		return 1;
	int r = 1;
	for(int i = 1; i <= m; ++i)
		r *= i;
	return r;
}

}


#endif
