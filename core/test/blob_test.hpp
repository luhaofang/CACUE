#include "../../mycnn.h"
#include "../utils/blob_ops.h"


#include <time.h>


using namespace mycnn;


void test_blob()
{
	blob * b = cacu_allocator::create_blob(1,3,224,224,test);
	//blob_ops::read_data2blob(b,"/home/seal/cuda-workspace/CACUE/core/test/python/feature_map.txt");
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
	for(int i = 1 ; i < 100; ++i)
	{
		b->resize(i,1,112,112);
		LOG_INFO("blob ptr: %ld",b);
		LOG_INFO("blob_data ptr: %ld",b->s_data());
		LOG_INFO("blob size: %d,%d,%d,%d",b->num(),b->channel(),b->width(),b->height());
		cuda_print(b->s_data()+b->count()-1, 1);
	}
	gettimeofday(&end, NULL);
	unsigned long diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	LOG_INFO("time cost:%ld ms", diff / 1000);
	delete b;
}
