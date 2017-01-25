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


#include "fft_conv.h"

using namespace mycnn;


int main(void) {

	int kernel_size = 5;
	int img_size = 56;

	fft_conv *fft = new fft_conv(kernel_size,img_size);

	float_t *img = new float_t[img_size*img_size];
	float_t *kernel = new float_t[kernel_size*kernel_size];
	float_t *res = new float_t[img_size*img_size];
	for(int i =0;  i < img_size*img_size ;i++)
		img[i] = i/img_size;
	for(int i = 0 ; i < kernel_size*kernel_size;i++)
		kernel[i] = i/kernel_size;
	for(int i  = 0; i < img_size*img_size ; i ++)
		res[i] = 0;
	clock_t start = clock();
	fft->cacu_fft_convolution(img,kernel,res);
	clock_t end = clock();
	printf("%d\n",end - start);
	//for(int i = 0 ; i < kernel_size*kernel_size; i ++)
	//	printf("%f,",res[i]);
}
