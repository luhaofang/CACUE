#include "../../mycnn.h"
#include <time.h>

#include "../graph/graph_ops.hpp"
#include "../graph/graph_factory.h"
using namespace mycnn;

void test_graph()
{


	network net;
	for (int i = 1; i <= 2; i++)
	{
		layer_block cs = create_block(i);
		net << cs;
	}

	graph_base *graph = graph_factory::create_graph();
	
    graph_phraser(net, graph);

	graph->run();

	printf("%d", net.length());

}
