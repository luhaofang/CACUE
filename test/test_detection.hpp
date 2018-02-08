#include "../example/mtcnn/demo.hpp"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"


TEST_CASE("detection")
{
	SECTION("detection functions test"){

		face_detect("/home/seal/4T/git/mtcnn-caffe/demo/test/56CA0642-7656-5D11-DE9E-5B564D21D32620170123_L.jpg");

	}

}

