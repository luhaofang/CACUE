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

#include <fftw3.h>

#include "factorize.h"

namespace mycnn{


	typedef float float_t;

	enum conv_type{same,unpadded};

	enum conv_fft_fresh{fresh,no_fresh};

	int FFTW_FACTORS[7] = {13,11,7,5,3,2,0}; // end with zero to detect the end of the array

	class fft_conv{

	public :

		fft_conv(int kernel_size, int img_size, int input_channel, int output_channel, float_t *w,conv_type type = same){

			_img_size = img_size;

			_kernel_size = kernel_size;

			_channel = input_channel;

			_output_channel = output_channel;

			_type = type;

			if(same == type){
				_fftw_size = find_closest_factor(_img_size + int(_kernel_size/2.0), FFTW_FACTORS);
				_out_img_size = _img_size;
			}
			else
			{
				_fftw_size = find_closest_factor(_img_size + _kernel_size - 1, FFTW_FACTORS);
			    _out_img_size = _img_size + _kernel_size - 1;
			}

			_in_src = new float_t[_fftw_size * _fftw_size];
			_out_src = (float_t*) fftwf_malloc(sizeof(fftwf_complex) * _fftw_size * (_fftw_size/2+1));
			_in_kernel = new float_t[_fftw_size * _fftw_size];
			_out_kernel = (float_t*) fftwf_malloc(sizeof(fftwf_complex) * _fftw_size * (_fftw_size/2+1));

			res_fft = new float_t[_fftw_size * _fftw_size];

			// Initialization of the plans
			p_forw_src = fftwf_plan_dft_r2c_2d(_fftw_size, _fftw_size, _in_src, (fftwf_complex*)_out_src, FFTW_ESTIMATE);
			p_forw_kernel = fftwf_plan_dft_r2c_2d(_fftw_size, _fftw_size, _in_kernel, (fftwf_complex*)_out_kernel, FFTW_ESTIMATE);

			fftwf_execute(p_forw_kernel);

			// The backward FFT takes out_kernel as input !!
			p_back = fftwf_plan_dft_c2r_2d(_fftw_size, _fftw_size, (fftwf_complex*)_out_kernel, res_fft, FFTW_ESTIMATE);
		};

		~fft_conv(){

			delete[] _in_src;
			fftwf_free((fftwf_complex*)_out_src);
			delete[] _in_kernel;
			fftwf_free((fftwf_complex*)_out_kernel);

			delete[] res_fft;

			// Destroy the plans
			fftwf_destroy_plan(p_forw_src);
			fftwf_destroy_plan(p_forw_kernel);
			fftwf_destroy_plan(p_back);
		};


		void cacu_fft_convolution(float_t *src, float_t *kernel , float_t *res,conv_fft_fresh fresh_)
		{
			float_t *ptr, *ptr_end, *ptr2 , *ptr_w;

			// Then we build our periodic signals
			//ptr = src;
			if(fresh == fresh_){
			// Reset the content of in_src
				for(ptr = _in_src, ptr_end = _in_src + _fftw_size*_fftw_size ; ptr != ptr_end ; ++ptr)
				  *ptr = 0.0;
				for(int i = 0 ; i < _img_size ; ++i)
				  for(int j = 0 ; j < _img_size ; ++j, ++ptr)
				  {
				_in_src[(i%_fftw_size)*_fftw_size+(j%_fftw_size)] += src[i*_img_size + j];
				  }
			}
			//ptr = kernel;

			for(ptr = _in_kernel, ptr_end = _in_kernel + _fftw_size*_fftw_size ; ptr != ptr_end ; ++ptr)
						  *ptr = 0.0;
			//rotato
			for(int i = 0 ; i < _kernel_size; ++i)
			{
				ptr_w = kernel + (_kernel_size - 1 - i) * _kernel_size;
			  for(int j = 0 ; j < _kernel_size; ++j, ++ptr,++ptr_w)
			  {
			_in_kernel[(i%_fftw_size)*_fftw_size+(j%_fftw_size)] += *ptr_w;
			  }
			}
			// And we compute their packed FFT
			fftwf_execute(p_forw_src);
			fftwf_execute(p_forw_kernel);

			// Compute the element-wise product on the packed terms
			// Let's put the element wise products in in_kernel
			float_t re_s, im_s, re_k, im_k;
			for(ptr = _out_src, ptr2 = _out_kernel, ptr_end = _out_src+2*_fftw_size * (_fftw_size/2+1); ptr != ptr_end ; ++ptr, ++ptr2)
			  {
				re_s = *ptr;
				im_s = *(++ptr);
				re_k = *ptr2;
				im_k = *(++ptr2);
				*(ptr2-1) = re_s * re_k - im_s * im_k;
				*ptr2 = re_s * im_k + im_s * re_k;
			  }

			// Compute the backward FFT
			// Carefull, The backward FFT does not preserve the output
			fftwf_execute(p_back);

			// Scale the transform
			for(ptr = res_fft, ptr_end = res_fft + _fftw_size*_fftw_size ; ptr != ptr_end ; ++ptr)
			{
			  *ptr /= float_t(_fftw_size*_fftw_size);
			}

			int _offset;

			if(same == _type)
			{
				_offset = int(_kernel_size/2.0);
				for(int i = 0 ; i < _out_img_size ; ++i)
					for(int j =0 ; j < _out_img_size ; ++j)
						res[i*_out_img_size + j] += res_fft[(i+_offset)*_fftw_size+_offset +j];
			}
			else
			{
				 for(int i = 0 ; i < _out_img_size ; ++i)
					 for(int j =0 ; j < _out_img_size ; ++j)
						res[i*_out_img_size + j] += res_fft[i*_fftw_size + j];
			}

		}

		void fft_transform(float_t *src, float_t *kernel , float_t *res)
		{

		}


	private :

		float_t *_in_src;

		float_t *_out_src;

		float_t *_in_kernel;

		float_t *_out_kernel;

		int _img_size;

		int _kernel_size;

		int _channel;

		int _output_channel;

		int _out_img_size;

		int _fftw_size;

		conv_type _type;

		float_t *res_fft;

		weight *_w;

		fftwf_plan p_forw_src;
		fftwf_plan p_forw_kernel;
		fftwf_plan p_back;
	};

};
