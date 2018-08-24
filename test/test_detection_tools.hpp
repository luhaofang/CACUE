#include "../cacu/cacu.h"
#include "../cacu/detection/detection.h"

#include <time.h>

using namespace cacu_detection;

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

TEST_CASE("detection_tools") {
	SECTION("detection functions test"){

	vector<rect *> *rects = new vector<rect *> ();
	set_rand_seed();
	for(int i = 0;i < 10; ++i) {
		rect *r1 = new rect(100 + i,300 + i,150 + i,400 + i, urand(0,1));
		LOG_INFO("%d,%d,%d,%d,%f", r1->l,r1->t,r1->r,r1->b,r1->score);
		rects->push_back(r1);
	}
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
	NMS(rects,0.6,nms_iou);
	gettimeofday(&end, NULL);
	unsigned long diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec
	- start.tv_usec;
	LOG_INFO("time cost:%ld ms", diff / 1000);
	LOG_INFO("rect length: %d", rects->size());
	for(int i = 0; i < rects->size(); ++i)
	LOG_INFO("%d,%d,%d,%d,%f", rects->at(i)->l,rects->at(i)->t,rects->at(i)->r,rects->at(i)->b,rects->at(i)->score);
	delete rects;
}

}
