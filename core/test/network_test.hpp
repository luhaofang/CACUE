#include "../../mycnn.h"
#include <time.h>

using namespace mycnn;

layer_block create_block(int i)
{

	layer *c = new layer(1, 1, 1, 1);

	blob *data = mycnn::cacu_allocator::create_blob(1, 100, 100, 100, 2, test);

	blobs *sblobs = mycnn::cacu_allocator::create_blobs();
	sblobs->push_back(data);

	layer *clayer = new layer(i, i, i, i, 1);

	c->op(CACU_INNERPRODUCT,data);

	layer_block cs(i, i, i, i, 1);
	cs << clayer << clayer << c;

	return cs;
}



network* create_alexnet()
{
	blob *b = cacu_allocator::create_blob(1, 3, 227, 227, 1, train);
	weight *_b = new weight("test",2, 3, 227, 227,train);
	_b->set_init_type(gaussian,1);
#if __PARALLELTYPE__ == __GPU__
	CUDA_PRINT(_b->s_data(),1);
#else
	LOG_INFO("%f,%f",_b->s_data()[0],_b->s_data()[1]);
#endif

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(_b);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_layer_maxpooling(_b, 96, 11, 4, 2);
	LOG_DEBUG("conv1");
	layer_block *conv2 = conv_layer_maxpooling((blob*)conv1->get_oblob(), 256, 5, 1, 2);
	LOG_DEBUG("conv2");
	layer_block *conv3 = conv_layer_nopooling((blob*)conv2->get_oblob(), 384, 3, 1, 1);
	LOG_DEBUG("conv3");
	layer_block *conv4 = conv_layer_nopooling((blob*)conv3->get_oblob(), 384, 3, 1, 1);
	LOG_DEBUG("conv4");
	layer_block *conv5 = conv_layer_maxpooling((blob*)conv4->get_oblob(), 256, 3, 1, 1);
	LOG_DEBUG("conv5");
	layer_block *fc6 = fc_layer((blob*)conv5->get_oblob(),4096);
	LOG_DEBUG("fc6");
	layer_block *fc7 = fc_layer((blob*)fc6->get_oblob(),4096);
	LOG_DEBUG("fc7");
	*net << conv1 << conv2 << conv3 << conv4 << conv5 << fc6 << fc7;

	return net;
}

void test_network()
{



}
