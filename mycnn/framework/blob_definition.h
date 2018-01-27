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


#ifndef __BLOBMAXREC__
#define __BLOBMAXREC__ 0XFFFFFFFF
#endif

/**
 * blob type definition, every blob_base maintains a blob type in order to turn back
 * original data maintaining type.
 */
enum blob_type{
	__blob__,
	__bin_blob__,
	__em_blob__,
	__em_bin_blob__
};


/**
 * definition of weights parameter initialization type.
 */
enum param_init_type {
	constant = 10,
	xavier = 11,
	gaussian = 12,
	msra = 13
};

/**
 * blob/network running mode.
 */
enum phase_type {
	test = 20,
	train = 21
};

/*
 * regularization type for weights update in optimization methods.
 *
 */
enum regularize_type {
	L1 = 30,
	L2 = 31
};

/*
 * regularization type for weights update in optimization methods.
 *
 */
enum normalize_type {
	norm_l1 = 40,
	norm_l2 = 41
};

typedef struct{
		int x;
		int y;
		int z;
}spatial3D;
