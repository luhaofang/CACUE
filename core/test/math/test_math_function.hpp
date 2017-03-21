#include "../../../mycnn.h"
#include "../../math/math_functions.h"
#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("gemv")
{


	SECTION("gemv_test"){

		blob *x = cacu_allocator::create_blob(1, 1, 3, 4, 1, test);

		blob *y = cacu_allocator::create_blob(1, 3, 1, 1, 0.2,test);

		blob *z = cacu_allocator::create_blob(1, 4, 1, 1, 0.2,test);

		cacu_print(x->s_data(),x->count());

		cacu_print(y->s_data(),y->count());

		cacu_sgemv(NOTRANS,x->s_data(),4,y->s_data(),3,1,z->s_data(),0);

		cacu_print(z->s_data(),z->count());


		delete x,y,z;

	}

}

TEST_CASE("gemm")
{


	SECTION("gemm_test"){

		blob *x = cacu_allocator::create_blob(1, 1, 3, 4, 1, test);

		blob *y = cacu_allocator::create_blob(1, 3, 1, 1, 0.2,test);

		blob *z = cacu_allocator::create_blob(1, 4, 1, 1, 0.2,test);

		cacu_print(x->s_data(),x->count());

		cacu_print(y->s_data(),y->count());

		cacu_sgemm(NOTRANS, NOTRANS,x->s_data(), 4, 3, y->s_data(),1,1,z->s_data(),0);

		cacu_print(z->s_data(),z->count());


		delete x,y,z;

	}

}
