#ifndef TEST_TENSOR_HPP_
#define TEST_TENSOR_HPP_



#include "../cacu/cacu.h"

#include "../cacu/framework/blob.h"
#include "../cacu/tensor/tensor.h"
#include "../cacu/utils/data_printer.h"
#include "../cacu/tensor/utils/device_data_utils.h"

#include <time.h>
#include "../tools/time_utils.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("tensor")
{
	SECTION("tensor functions test"){

		tensor<float> t(1);
		vec_t a(1,3);
		t.copy2data(&a[0]);
		cacu_print(t.pdata(),1);
		LOG_DEBUG("p initial: %f",t.pdata()[0]);

		t.resize(3, 1);
		LOG_DEBUG("p resize: %f",t.pdata()[0]);
		LOG_DEBUG("tensor length:%d", t.length());
		a.resize(3,2);
		cacu_print_cpu(&a[0],3);
		t.copy2data(&a[0]);
		cacu_print(t.pdata(),3);
		t.refresh();
		cacu_print(t.pdata(),3);

	}

	SECTION("resize costs"){
		tensor<float> t(3*3*224*224);
		for(int i = 1 ; i < 1; i++)
		{
			t.resize(i*3*224*224, 1);
		}
		LOG_INFO("this is a test");
	}
}



#endif
