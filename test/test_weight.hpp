#include "../cacu/cacu.h"

#include "../cacu/framework/weight.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("weight") {
	SECTION("weight functions test"){

	weight w("test", 1, 3, 3, 3, test);

	w.set_init_type(gaussian, 0.1);
	cacu_print(w.s_data(),27);

	w.set_init_type(constant, 0.1);
	cacu_print(w.s_data(),27);

	w.set_init_type(xavier, 0.1);
	cacu_print(w.s_data(),27);

	w.set_init_type(msra, 0.1);
	cacu_print(w.s_data(),27);

}

}

