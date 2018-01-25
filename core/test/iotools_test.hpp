#include "../../tools/imageio_utils.h"
#include <time.h>
#include <sys/time.h>

using namespace mycnn_tools;

void test_iotools()
{
	blob* b = new blob(1,3,256,256,0,test);
	weight *_b = new weight("test",1, 3, 224, 224,test);
	_b->set_init_type(gaussian,1);
	set_rand_seed();
	//imageio_utils::save_mean_file(_b->s_data(),"/home/seal/test_quick.model",3*224*224);
	//imageio_utils::load_mean_file(b->s_data(),"/home/seal/test_quick.model");


	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
	for(int i = 0; i < 100; ++i)
#if __PARALLELTYPE__ == __CUDA__
		imageio_utils::clip_imread_gpu(_b->s_data(), "/home/seal/n01440764_18.JPEG", 224, 224);
#else
		imageio_utils::clip_imread(_b->s_data(), "/home/seal/n01440764_18.JPEG", 224, 224);
#endif
	gettimeofday(&end, NULL);
	long diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
						- start.tv_usec;
	LOG_INFO("%ld ms/iter", diff);

	gettimeofday(&start, NULL);
	for(int i = 0; i < 100; ++i)
#if __PARALLELTYPE__ == __CUDA__
		imageio_utils::imread_gpu(_b->s_data(), "/home/seal/n01440764_18_.JPEG", 224*224*3);
#else
		imageio_utils::imread(_b->s_data(), "/home/seal/n01440764_18_.JPEG", 224*224*3);
#endif
	gettimeofday(&end, NULL);
	diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
						- start.tv_usec;
	LOG_INFO("%ld ms/iter", diff);

}
