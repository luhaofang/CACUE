#include "../../tools/imageio_utils.h"


using namespace mycnn_tools;

void test_iotools()
{
	blob* b = new blob(1,3,224,224,0,test);
	weight *_b = new weight("test",1, 3, 224, 224,test);
	_b->set_init_type(gaussian,1);
	imageio_utils::save_mean_file(_b->s_data(),"/home/seal/test_quick.model",3*224*224);
	imageio_utils::load_mean_file(b->s_data(),"/home/seal/test_quick.model");
}
