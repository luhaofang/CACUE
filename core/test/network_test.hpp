#include "../../mycnn.h"
#include <time.h>

using namespace mycnn;

layer_block create_block(int i)
{

	layer *c = new layer(1, 1, 1, 1);

	blob *data = mycnn::cacu_allocator::create_blob(1, 100, 100, 100, 2);

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
	
	network *net = new network();

	layer_block *conv1 = conv_layer_maxpooling(b, 96, 11, 4, 2);
	layer_block *conv2 = conv_layer_maxpooling((blob*)conv1->get_oblob(), 256, 5, 1, 2);
	layer_block *conv3 = conv_layer_nopooling((blob*)conv2->get_oblob(), 384, 3, 1, 1);
	layer_block *conv4 = conv_layer_nopooling((blob*)conv3->get_oblob(), 384, 3, 1, 1);
	layer_block *conv5 = conv_layer_maxpooling((blob*)conv4->get_oblob(), 256, 3, 1, 1);

	*net << conv1 << conv2 << conv3 << conv4 << conv5;

	return net;
}

void test_network()
{

	network net;
	for (int i = 1; i <= 2;i++)
	{
		layer_block cs = create_block(i);
		net << cs;
	}

	printf("%d",net.length());


}
