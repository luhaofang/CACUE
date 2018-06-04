#include "../example/mtcnn/demo.hpp"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"


TEST_CASE("detection")
{
	SECTION("detection functions test"){

		face_detect("/home/haofang/git/Detectron/demo/test.jpg");

	}

}

