/*
 * test_cacu_op.hpp
 *
 *  Created on: May 14, 2018
 *      Author: haofang
 */

#ifndef TEST_PEARSON_HPP_
#define TEST_PEARSON_HPP_

#include <time.h>

#include "../cacu/framework/network.h"
#include "../cacu/framework/gframework/cacu_graphic.h"


#include "../example/imagenet/resnet_18.h"
#include "../example/cifar10/cifar_quick_net.h"

#define CATCH_CONFIG_MAIN
#include "../3rdparty/catch.hpp"

using namespace cacu;

void pearson(weight *&w_){
	if(w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());

	float_t all_pearson = 0;

	ostringstream oss;

	int count = 0;
	for(int i = 0; i < w_->num(); ++i){
		float m = 0, s = 0;
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		m = m / w_->length();
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr(&temp[0], w_->length(), &temp[0]);
		cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
		s = sqrtf(s / (w_->length()));
		for(int j = i + 1; j < w_->num(); ++j){
			float mj = 0, sj = 0, mjm = 0;
			cacu_sumbysize(BYWIDTH, w_->p_data(j), w_->length(), (float_t)1, &mj, (float_t)0, w_->length());
			mj = mj / w_->length();
			cacu_sdxsize(w_->p_data(j), w_->length(), (float_t)-1 * mj, (float_t)1, &temp[0]);
			cacu_sqr(&temp[0], w_->length(), &temp[0]);
			cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &sj, (float_t)0, w_->length());
			sj = sqrtf(sj / (w_->length()));

			cacu_copy(w_->p_data(i),w_->length(), &temp[0]);
			cacu_ssx(w_->p_data(j), w_->length(), &temp[0]);
			cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &mjm, (float_t)0, w_->length());
			mjm = mjm / w_->length();

			mjm = (mjm - m*mj) / (s * sj);
			LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f, %.8f]", m, s, mj, sj, mjm);
			count += 1;
			all_pearson += abs(mjm);
			oss << mjm << ",";
		}
	}
	LOG_INFO("%.8f", all_pearson / count);
	LOG_INFO("%s", oss.str().c_str());
	//cacu_print(w_->s_data(), w_->count());
}

void cosine(weight *&w_){
	if(w_->length() == 1)
		return;
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());

	float_t all_pearson = 0;

	ostringstream oss;

	int count = 0;
	for(int i = 0; i < w_->num(); ++i){
		float m = 0, s = 0;
		cacu_sqr(w_->p_data(i), w_->length(), &temp[0]);
		cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &m, (float_t)0, w_->length());
		s = sqrtf(m);
		for(int j = i + 1; j < w_->num(); ++j){
			float mj = 0, sj = 0, mjm = 0;
			cacu_sqr(w_->p_data(j), w_->length(), &temp[0]);
			cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &mj, (float_t)0, w_->length());
			sj = sqrtf(mj);

			cacu_copy(w_->p_data(i),w_->length(), &temp[0]);
			cacu_ssx(w_->p_data(j), w_->length(), &temp[0]);
			cacu_sumbysize(BYWIDTH, &temp[0], w_->length(), (float_t)1, &mjm, (float_t)0, w_->length());
			mjm = mjm / (s * sj);

			LOG_DEBUG("[%.8f, %.8f, %.8f, %.8f, %.8f]", m, s, mj, sj, mjm);
			count += 1;
			all_pearson += abs(mjm);
			oss << mjm << ",";
		}
	}
	LOG_INFO("%.8f", all_pearson / count);
	LOG_INFO("%s", oss.str().c_str());
	//cacu_print(w_->s_data(), w_->count());
}

TEST_CASE("pearson") {
	SECTION("network test"){
//		blobs *inputs = new blobs();
//		inputs->push_back(new blob(1, 3, 32, 32, 0, test));
//		network *net = phrase_network("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test.ch", inputs);
		network *net = create_cifar_quick_net(1,test);
		LOG_DEBUG("fuck!");
		net->set_is_use_bias(false);

		net->load_weights("/Users/seallhf/Documents/datasets/cifar10/cifar10_quick_test_77.26.model");
		net->weights_pre_proc(pearson);
	}
}






#endif /* TEST_DECONV_HPP_ */
