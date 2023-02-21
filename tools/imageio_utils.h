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
#include "../cacu/framework/blob.h"

#include "../tools/string_utils.h"
#include "../cacu/tensor/cuda/cuda_utils.h"
#include "../cacu/math/utils/rand_t.h"
#include "../cacu/utils/log.h"
#include "../cacu/utils/check_utils.h"

using namespace std;
using namespace cacu;


namespace cacu_tools {

	/**
	 *
	 * using opencv to finish image reading process,
	 * Data organized by [BGR] channels
	 */
	class imageio_utils {

	public:

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
		static void imread_gpu(cacu::float_t *p_data, const char* file_path_, const int size);

		static cacu::blob* read_img2blob_gpu(const char* file_path_, phase_type phase_);

		static void resize_imread_gpu(cacu::float_t *p_data, const char* file_path_, int resize_h, int resize_w);

		static void clip_imread_gpu(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, bool flip, cacu::float_t *mean=NULL);

		static void center_clip_imread_gpu(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, cacu::float_t *mean=NULL);

		static void imwrite_gpu(blob *blob_, const char* file_path_);

#endif
#endif

		static void imread_cpu(cacu::float_t *p_data, const char* file_path_, const int p_size);

		static cacu::blob* read_img2blob_cpu(const char* file_path_, phase_type phase_);

		static void resize_imread_cpu(cacu::float_t *p_data, const char* file_path_, int resize_h, int resize_w);

		static void clip_imread_cpu(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, bool flip, cacu::float_t *mean=NULL);

		static void center_clip_imread_cpu(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, cacu::float_t *mean=NULL);

		static void imwrite_cpu(blob *blob_, const char* file_path_);

		//*/

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
		static void load_mean_file_gpu(cacu::float_t *p_data, string mean_file_);
#endif
#endif
		static void save_mean_file(cacu::float_t *p_data, const string& mean_file_, int length_);

		static void load_mean_file(cacu::float_t *p_data, const string& mean_file_);

		static void resize_imread(cacu::float_t *p_data, const char* file_path_, int resize_h, int resize_w);

		static void imwrite(blob *blob_, const char* file_path_);

		static void imread(cacu::float_t *p_data, const char* file_path_, const int p_size);

		static void clip_imread(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, bool flip);

		static void center_clip_imread(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w);

		static void clip_imread_channel_wise_norm(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, const vec_t &channels, bool flip = true);

		static void center_clip_imread_channel_wise_norm(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, const vec_t &channels);

		static void imread_channel_wise_norm(cacu::float_t *p_data, const char* file_path_, int p_size, const vec_t &channels);

		static void clip_imread_element_wise_norm(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, cacu::float_t *mean, bool flip = true);

		static void center_clip_imread_element_wise_norm(cacu::float_t *p_data, const char* file_path_, int clip_size_h, int clip_size_w, cacu::float_t *mean);

		static cacu::blob* read_img2blob(const char* file_path_, phase_type phase_);
	};
}


#endif
