#ifndef TEST_CUBLAS_HPP_
#define TEST_CUBLAS_HPP_


#include "../cacu/cacu.h"

#include "../cacu/framework/blob.h"
#include "../cacu/tensor/tensor.h"
#include "../cacu/utils/data_printer.h"
#include "../cacu/tensor/utils/device_data_utils.h"

#include "../cacu/math/math_functions.h"

#include "../cacu/math/blas/cublas_utils.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("cublas")
{
	SECTION("cublas functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
		set_rand_seed();

		create_cublas_handle();

		weight *w1 = new weight("a",1,1,1,10,test);
		weight *w2 = new weight("b",1,1,1,10,test);

		w1->set_init_type(msra,1.0);
		w2->set_init_type(msra,2.0);

		cacu_print(w1->s_data(),10);
		cacu_print(w2->s_data(),10);

		cacu_saxpby(w1->s_data(),1,w2->s_data(),2,w1->count());
		cacu_print(w2->s_data(),10);

		release_cublas_handle();

		delete w1;
		delete w2;
#endif
#endif
	}


	SECTION("blas functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
		set_rand_seed();

		create_cublas_handle();

		weight *w1 = new weight("a",1,3,5,5,test);
		weight *w2 = new weight("b",1,3,5*3,5*3,test);

		w1->set_init_type(msra,1.0);
		w2->set_init_type(msra,2.0);

		cacu_print(w1->s_data(),10);
		cacu_print(w2->s_data(),10);

		cacu_img2col_pad(w1->s_data(),3,3,1,5,5,3,5,5,1,1,w2->s_data());
		cacu_print(w2->s_data(),10);

		release_cublas_handle();

		delete w1;
		delete w2;
#endif
#endif

	}
}

TEST_CASE("blas")
{
	SECTION("cublas functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
		set_rand_seed();

		create_cublas_handle();

		weight *w1 = new weight("a",1,1,1,10,test);
		weight *w2 = new weight("b",1,1,1,10,test);

		w1->set_init_type(msra,1.0);
		w2->set_init_type(msra,2.0);

		cacu_print(w1->s_data(),10);
		cacu_print(w2->s_data(),10);

		cacu_saxpby(w1->s_data(),1,w2->s_data(),2,w1->count());
		cacu_print(w2->s_data(),10);

		release_cublas_handle();

		delete w1;
		delete w2;
#else
		weight *w1 = new weight("a",1,2,1,3,test);
		weight *w2 = new weight("b",1,6,1,2,test);
		weight *w3 = new weight("b",1,1,1,2,test);


		//w1->set_init_type(msra,1.0);
		//w2->set_init_type(msra,2.0);

		vec_t a(6);
		a[0] = 0.1215;
		a[1] = -1.512;
		a[2] = -0.035;
		a[3] = 0.5532;
		a[4] = 0.00153;
		a[5] = 0.3435;
		vec_t b(12);
		b[0] = 0.3664;
		b[1] = -0.0253;
		b[2] = 0.102053;
		b[3] = -0.7754;
		b[4] = 0.66732;
		b[5] = 0.0355;
		b[6] = 1.2562;
		b[7] = -0.6325;
		b[8] = 0.0352;
		b[9] = 0.110225;
		b[10] = 0.02053;
		b[11] = -0.02534;
		w1->copy2data(a);
		w2->copy2data(b);
		cacu_print(w1->s_data(),w1->count());
		cacu_print(w2->s_data(),w2->count());

		cacu_sgemv(NOTRANS, w2->s_data(), 2, w1->s_data(), 6, (float_t)1., w3->s_data(),(float_t)0);
		cacu_print(w3->s_data(), w3->count());
		cacu_l2_normalization(w3->s_data(), w3->count(), 1e-12);

		cacu_print(w3->s_data(), w3->count());

		cacu_sgemv(NOTRANS, w2->s_data(), 2, w1->s_data(), 6, (float_t)1., w3->s_data(),(float_t)0);
		cacu_print(w3->s_data(),w3->count());

		cacu_sgemm(NOTRANS, NOTRANS, w2->s_data(), 2, 6, w1->s_data(), 1, (float_t)1., w3->s_data(),(float_t)0);
		cacu_print(w3->s_data(),w3->count());

#endif
#endif
	}
}

TEST_CASE("blas_random")
{
	SECTION("cublas functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
		cuda_set_device(0);
		set_rand_seed();

		create_cublas_handle();

		weight *w1 = new weight("a",1,1,1,10,test);
		weight *w2 = new weight("b",1,1,1,10,test);

		w1->set_init_type(msra,1.0);
		w2->set_init_type(msra,2.0);

		cacu_print(w1->s_data(),10);
		cacu_print(w2->s_data(),10);

		cacu_saxpby(w1->s_data(),1,w2->s_data(),2,w1->count());
		cacu_print(w2->s_data(),10);

		release_cublas_handle();

		delete w1;
		delete w2;
#else
		weight *w1 = new weight("a",1,3,3,3,test);
		weight *w2 = new weight("b",1,4,3,3,test);
		weight *w3 = new weight("b",3,4,1,1,test);

		w1->set_init_type(gaussian,1);
		w2->set_init_type(gaussian,1);

		cacu_print(w1->s_data(),w1->count());
		cacu_print(w2->s_data(),w2->count());

		cacu_sgemm(NOTRANS, NOTRANS, w1->s_data(), w1->channel(), w1->channel_length(), w2->s_data(), w2->channel(), (float_t)1., w3->s_data(),(float_t)0);
		cacu_print(w3->s_data(),w3->count());
		//w1->resize(1,3,3,1);
		cacu_sgemm(TRANS, NOTRANS, w1->s_data(), w1->channel(), w1->channel_length(), w2->s_data(), w2->channel(), (float_t)1., w3->s_data(),(float_t)0);
		cacu_print(w3->s_data(),w3->count());
		cacu_sgemm(NOTRANS, TRANS, w1->s_data(), w1->channel(), w1->channel_length(), w2->s_data(), w2->channel(), (float_t)1., w3->s_data(),(float_t)0);
		cacu_print(w3->s_data(),w3->count());
		cacu_sgemm(TRANS, TRANS, w1->s_data(), w1->channel(), w1->channel_length(), w2->s_data(), w2->channel(), (float_t)1., w3->s_data(),(float_t)0);
		cacu_print(w3->s_data(),w3->count());

		cacu_sgemv(NOTRANS, w1->s_data(), w1->channel(), w2->s_data(), w1->channel_length(), (float_t)1., w3->s_data(), (float_t)0);
		cacu_print(w3->s_data(),w3->count());
		cacu_sgemv(TRANS, w1->s_data(), w1->channel(), w2->s_data(), w1->channel_length(), (float_t)1., w3->s_data(), (float_t)0);
		cacu_print(w3->s_data(),w3->count());

#endif
#endif
	}
}



#endif
