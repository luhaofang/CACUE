#include "../../../mycnn.h"
#include "../../detect/detection.h"


#include <time.h>

using namespace mycnn_tools;
using namespace mycnn_detection;


void test_tools()
{
	vector<rect *> rects;
	set_rand_seed();
	for(int i = 0;i < 10; ++i){
		rect *r1 = new rect(urandint(0,100),urandint(0,300),urandint(50,150),urandint(200,400), urand(0,1));
		LOG_INFO("%d,%d,%d,%d,%f", r1->l,r1->t,r1->r,r1->b,r1->score);
		rects.push_back(r1);
	}
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
	NMS(rects,0.7,nms_iou);
	gettimeofday(&end, NULL);
	unsigned long diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
						- start.tv_usec;
	LOG_INFO("time cost:%ld ms", diff / 1000);
	LOG_INFO("rect length: %d", rects.size());
	for(int i = 0 ; i < rects.size(); ++i)
		LOG_INFO("%d,%d,%d,%d,%f", rects[i]->l,rects[i]->t,rects[i]->r,rects[i]->b,rects[i]->score);
}
