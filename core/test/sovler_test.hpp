#include "../../mycnn.h"
#include <time.h>

using namespace mycnn;


void test_sovler()
{
	network *net = create_alexnet();
	sgd_solver *_solver = new sgd_solver(net);
	_solver->echo();
	//LOG_INFO("_solver size: %d",);
}
