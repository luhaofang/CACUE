#include "../../../mycnn.h"
#include "../../math/math_functions.h"
#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../../../3rd_party/catch.hpp"

using namespace mycnn;

TEST_CASE("random")
{


	SECTION("random_test"){

		set_rand_seed();

		cout << "urand:";
		for(int i = 0; i < 100; ++i)
			cout << urand(0.0,1.0) << ",";
		cout << endl;

		cout << "gaussian:";
		for(int i = 0; i < 1000; ++i)
			cout << gaussrand(2.0) << ",";
		cout << endl;

	}

}

TEST_CASE("random_vector")
{


	SECTION("random_test"){

		set_rand_seed();

		blob *x = cacu_allocator::create_blob(1, 100, 1, 1, 1, test);

		blob *y = cacu_allocator::create_blob(1, 100, 1, 1, 1,test);

		rand_vector(x->s_data(),100,0.1);
		rand_vector(y->s_data(),100,0.1);
		cacu_print(x->s_data(),100);
		cacu_print(y->s_data(),100);


	}

}
