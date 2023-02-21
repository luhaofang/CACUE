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

#ifndef MATH_FUNCTION_NAIVE_H_
#define MATH_FUNCTION_NAIVE_H_


#include "../../config.h"

#include "../math_definition.h"

#if __CBLASTYPE__ == __NAIVE__

namespace cacu{

inline void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc) {
    int i,j,k;
    float A_PART;
#if __OPENMP__ == ON
    #pragma omp parallel for default(shared) private(i,j,k,A_PART)
#endif
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
        	A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[j*ldc+i] += A_PART*B[k*ldb+j];
            }
        }
    }
}

inline void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc) {
    int i,j,k;
    float sum;
#if __OPENMP__ == ON
    #pragma omp parallel for default(shared) private(i,j,k,sum)
#endif
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[j*ldc+i] += sum;
        }
    }
}

inline void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc) {
    int i,j,k;
    float A_PART;
#if __OPENMP__ == ON
    #pragma omp parallel for default(shared) private(i,j,k,A_PART)
#endif
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[j*ldc+i] += A_PART*B[k*ldb+j];
            }
        }
    }
}

inline void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc) {
    int i,j,k;
    float sum;
#if __OPENMP__ == ON
    #pragma omp parallel for default(shared) private(i,j,k,sum)
#endif
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[j*ldc+i] += sum;
        }
    }
}


/**
 * CACUE use column major storage for blas
 */

inline void gemm_cpu(TRANSPOSE TA, TRANSPOSE TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc) {
    int i, j;
    for(i = 0; i < M; ++i) {
        for(j = 0; j < N; ++j) {
            C[j*ldc + i] *= BETA;
        }
    }
    if(TA == NOTRANS && TB == NOTRANS)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if(TA == TRANS && TB == NOTRANS)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if(TA == NOTRANS && TB == TRANS)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

inline void cacu_sgemm_naive(TRANSPOSE transx_, TRANSPOSE transy_, float *x,const int x_height,
		const int x_width,float *y,const int y_width,const float alpha,float *z,const float beta) {
	int m = x_height,n = y_width,k = x_width;
	int lda = (transx_ == NOTRANS) ? m : k;
	int ldb = (transy_ == NOTRANS) ? k : n;
    gemm_cpu(transx_, transy_, m, n, k, alpha, x, lda, y, ldb, beta, z, m);
}

inline void cacu_sgemv_naive(TRANSPOSE trans_, float *x, const int x_height, float *y,
		const int x_width, const float alpha, float *z, const float beta) {
	int m = (trans_ == NOTRANS) ? x_height : x_width;
	int k = (trans_ == NOTRANS) ? x_width : x_height;
	int lda = (trans_ == NOTRANS) ? m : k;
    gemm_cpu(trans_, NOTRANS, m, 1, k, alpha, x, lda, y, k, beta, z, m);
}

inline void cacu_saxpy_naive(float *x, const float a, float *y, const int length) {
	int i;
#if __OPENMP__ == ON
	#pragma omp parallel for
#endif
	for(i = 0; i < length; ++i)
		y[i] = a*x[i] + y[i];
}

inline void cacu_saxpby_naive(float *x, const float a, float *y, const float b, const int length) {
	int i;
#if __OPENMP__ == ON
	#pragma omp parallel for
#endif
	for(i = 0; i < length; ++i)
		y[i] = a*x[i] + b*y[i];
}

inline void cacu_scalex_naive(float *x, const float a, const int length) {
	int i;
#if __OPENMP__ == ON
	#pragma omp parallel for
#endif
	for(i = 0; i < length; ++i)
		x[i] = a*x[i];
}

inline void cacu_copy_naive(const float *x, int length, float *y) {
	int i;
#if __OPENMP__ == ON
	#pragma omp parallel for
#endif
	for(i = 0; i < length; ++i)
		y[i] = x[i];
}

}

#endif

#endif
