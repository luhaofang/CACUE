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

#ifndef IMAGEIO_UTILS_H_
#define IMAGEIO_UTILS_H_

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;


#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "../cacu/definition.h"

#include "../tools/string_utils.h"
#include "../cacu/tensor/cuda/cuda_utils.h"
#include "../cacu/math/utils/rand_t.h"

using namespace std;
using namespace cacu;


namespace cacu_tools {


	class imageio_utils {

	public:

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
		static void imread_gpu(cacu::float_t *p_data, const char* file_path_, const int size);

		static void resize_imread_gpu(cacu::float_t *p_data, const char* file_path_, int resize_h, int resize_w);

		static void clip_imread_gpu(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w);
#endif
#endif

		static void imread(cacu::float_t *p_data, const char* file_path_, const int p_size);

		static void resize_imread(cacu::float_t *p_data, const char* file_path_, int resize_h, int resize_w);

		static void clip_imread(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w);
		//*/

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
		static void load_mean_file_gpu(cacu::float_t *p_data, string mean_file_);
#endif
#endif
		static void save_mean_file(cacu::float_t *p_data, string mean_file_, int length_);

		static void load_mean_file(cacu::float_t *p_data, string mean_file_);

	};
}


#endif
