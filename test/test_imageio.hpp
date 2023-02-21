#include "../tools/imageio_utils.h"
#include "../example/frcnn/tools_frcnn.h"
#include "../cacu/math/math_functions.h"

#include <time.h>

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"
#include "../cacu/cacu.h"
using namespace cacu_tools;
using namespace cacu;

TEST_CASE("transpose")
{
	SECTION("imageio functions test"){

		blob *b = new blob(4,1,4,4,0,test);

		b->set_init_type(gaussian, 1);

//		cacu_print(b->s_data(),b->count());

		cacu_transpose(b->s_data(),4,1,16);
//		cacu_print(b->s_data(),b->count());
		cacu_transpose(b->s_data(),1,4,16);

		cacu_print(b->p_data(0),b->length());


		cacu_rotate(b->p_data(0), 4, 4, 1, rotate_left);
		cacu_print(b->p_data(0), b->length());

		cacu_rotate(b->p_data(0), 4, 4, 1, rotate_bottom);
		cacu_print(b->p_data(0), b->length());

		cacu_rotate(b->p_data(0), 4, 4, 1, rotate_right);
		cacu_print(b->p_data(0), b->length());
		//delete b;

	}

}

TEST_CASE("readimg")
{
	SECTION("imageio functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	set_rand_seed();
	blob *b = new blob(1,3,224,224,255,test);

	string filepath = "/home/seallhf/57.jpg";
	resize_readimg(b->s_data(),filepath.c_str(),224,224);

	blobs *sblob = new blobs();
	sblob->push_back(b);
	LOG_DEBUG("fuck");
	data_args *args = new data_args(3, 2, 2, 0, 3);
	un_pooling_op *op = new un_pooling_op(sblob, (data_args *&)args);
	op->infer();
	imageio_utils::imwrite(((blob*)op->out_data<blob>()), "/home/seallhf/57_unpooling.jpg");

	//delete b;


	}

}


TEST_CASE("blur")
{
	SECTION("imageio functions test"){

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	set_rand_seed();
	blob *b = new blob(1,3,224,224,255,train);

	string filepath = "/home/seallhf/FC_l8_34.jpg";
	resize_readimg(b->s_data(),filepath.c_str(),224,224);

	blobs *sblob = new blobs();
	sblob->push_back(b);
	LOG_DEBUG("fuck");
	data_args *args = new data_args(3, 2, 2, 0, 3);
	blur_op *op = new blur_op(sblob, (data_args *&)args);
	op->infer();
	op->out_data<blob>()->set_diff(1);
	op->derivative();
	cacu_print(b->s_diff(), b->count());
	imageio_utils::imwrite(((blob*)op->out_data<blob>()), "/home/seallhf/test.jpg");

	//delete b;


	}

}
