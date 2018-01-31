#include "../cacu/cacu.h"

#include "../cacu/framework/blob.h"
#include "../cacu/tensor/tensor.h"
#include "../cacu/utils/data_printer.h"
#include "../cacu/tensor/utils/device_data_utils.hpp"

#include "../cacu/math/math_functions.hpp"

#include "../cacu/math/blas/cublas_utils.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

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
#endif
#endif
	}


	SECTION("blas functions test"){

		weight *w1 = new weight("a",1,1,1,10,test);
		weight *w2 = new weight("b",1,1,1,10,test);

		w1->set_init_type(constant,1.0);
		w2->set_init_type(constant,2.0);

		cacu_print(w1->s_data(),10);
		cacu_print(w2->s_data(),10);

		cacu_saxpby(w1->s_data(),1,w2->s_data(),1,w1->count());
		cacu_print(w2->s_data(),10);

		delete w1;
		delete w2;

	}
}

