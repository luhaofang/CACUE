#ifndef TEST_LOG_HPP_
#define TEST_LOG_HPP_

#include "../cacu/cacu.h"

#include "../cacu/utils/log.h"
#include "../cacu/utils/check_utils.h"
#include "../tools/imageio_utils.hpp"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;
using namespace cacu_tools;

TEST_CASE("utils")
{
	SECTION("log functions test") {


		LOG_INFO("this just a test %d, %d",1,12);
		LOG_DEBUG("this just a test %d, %d", 1, 12);
		//LOG_FATAL("this just a test %d, %d", 1, 12);
		LOG_CHECK("this just a test %d, %d", 1, 12);


		//CHECK_EQ_OP(1, 2, "this just a test %d,%d", 1, 2);
		//CHECK_GE_OP(1, 2, "this just a test %d,%d", 1, 2);
		//CHECK_GT_OP(1, 2, "this just a test %d,%d", 1, 2);
		//CHECK_LE_OP(2, 1, "this just a test %d,%d", 2, 1);
		
	}

}



#endif