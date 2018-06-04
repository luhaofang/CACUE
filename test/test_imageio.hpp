#include "../tools/imageio_utils.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"
using namespace cacu_tools;

TEST_CASE("imageio")
{
	SECTION("imageio functions test"){

		blob *b = new blob(4,3,224,224,255,test);

		string filepath = "/home/haofang/test.jpg";
		imageio_utils::imwrite(b->s_data(),filepath.c_str(),b->num(),b->width(),b->height());

		//delete b;

	}

}

