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


#include "cuda_init_utils.h"

#include "../../config.h"
#include "../../math/blas/cublas_utils.h"

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

namespace cacu {

cublasHandle_t handle = NULL;

void create_cublas_handle(){
	CUBLAS_CHECK(cublasCreate_v2(&handle));
}

void release_cublas_handle() {
	CUBLAS_CHECK(cublasDestroy_v2(handle));
}

void cuda_set_device(unsigned int device_id) {
	struct cudaDeviceProp device_prop;
	if (cudaGetDeviceProperties(&device_prop, device_id) == cudaSuccess) {


		cout << "======================================================="
				<< endl;
		cout << "device " << device_id << ": " << device_prop.name << endl;
		cout << "-------------------------------------------------------"
				<< endl;
		cout << "totalGlobalMem      |	" << device_prop.totalGlobalMem << endl;
		cout << "warpSize            |	" << device_prop.warpSize << endl;
		cout << "maxThreadsPerBlock  |	" << device_prop.maxThreadsPerBlock
				<< endl;
		cout << "sharedMemPerBlock   |	" << device_prop.totalConstMem << endl;
		cout << "totalConstMem       |	" << device_prop.totalConstMem << endl;
		cout << "======================================================="
				<< endl;
	} else
		cout << "device " << device_id
				<< " not found, please check your device num or select an available device!" << endl;
	CUDA_CHECK(cudaSetDevice(device_id));
	create_cublas_handle();
}

void cuda_release() {
	release_cublas_handle();
}

void cuda_free(void* data_) {
	cudaFree(data_);
}


}

#endif
#endif
