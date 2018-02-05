#include "../cacu/cacu.h"

#include "../cacu/framework/blob.h"
#include "../cacu/tensor/tensor.h"
#include "../cacu/utils/data_printer.h"
#include "../cacu/tensor/utils/device_data_utils.hpp"

#include <time.h>

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
		LOG_DEBUG("p initial: %d",t.pdata());

		t.resize(3, 1);
		LOG_DEBUG("p resize: %d",t.pdata());
		LOG_DEBUG("tensor length:%d", t.length());
		a.resize(3,2);
		cacu_print_cpu(&a[0],3);
		t.copy2data(&a[0]);
		cacu_print(t.pdata(),3);
		t.refresh();
		cacu_print(t.pdata(),3);

	}

	SECTION("resize costs"){
		struct timeval start;
		struct timeval end;
		unsigned long diff;
		tensor<float> t(3*3*224*224);
		gettimeofday(&start, NULL);
		for(int i = 1 ; i < 1; i++)
		{
			t.resize(i*3*224*224, 1);
		}
		gettimeofday(&end, NULL);
		diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
							- start.tv_usec;
		LOG_INFO("resize time cost %ld ms/iter",diff / 1000);
	}
}

