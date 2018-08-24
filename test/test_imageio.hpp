#include "../tools/imageio_utils.h"
#include "../example/frcnn/tools_frcnn.h"
#include "../cacu/math/math_functions.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"
using namespace cacu_tools;

TEST_CASE("transpose")
{
	SECTION("imageio functions test"){

		blob *b = new blob(4,3,3,3,0,test);


		b->set_pdata(0,0);
		b->set_pdata(1,1);
		b->set_pdata(2,2);
		b->set_pdata(3,3);
		cacu_print(b->s_data(),b->count());

		cacu_transpose(b->s_data(),4,3,9);
		cacu_print(b->s_data(),b->count());
		cacu_transpose(b->s_data(),3,4,9);

		cacu_print(b->s_data(),b->count());

		//delete b;

	}

}

TEST_CASE("readimg")
{
	SECTION("imageio functions test"){

		blob *b = new blob(1,3,224,224,255,test);

		string filepath = "/home/haofang/data/faces/Emile Hirsch/Emile Hirsch_18967.jpg";
		resize_readimg(b->s_data(),filepath.c_str(),224,224);

		//delete b;

	}

}
