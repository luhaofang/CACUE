/*
 Copyright (c) 2016, David lu
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of the <organization> nor the
 names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef TRAIN_CYCLE_GAN_HPP_
#define TRAIN_CYCLE_GAN_HPP_

#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"
//#include "../../cacu/solvers/sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../frcnn/tools_frcnn.h"

#include "../imagenet/data_proc.h"
#include "models.h"

using namespace cacu;

void train_net() {
	int batch_size = 1;

	int max_iter = 100000;

	int test_iter = 100;
	int train_test_iter = 100;
	int img_size = 256;

	string root_path = "/home/seallhf/4T/dataset/cycleGAN/datasets/horse2zebra/";
	//string root_path = "/data1/luhaofang/datasets/datasets/horse2zebra/";
	//set gpu device if training by gpu
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_set_device(0);
#endif
#endif
	//set random seed
	set_rand_seed();

	//log output
	std::ofstream xdlog(root_path + "xdiscriminator.log",
			ios::binary);
	xdlog.precision(std::numeric_limits<cacu::float_t>::digits10);
	//log output
	std::ofstream ydlog(root_path + "ydiscriminator.log",
			ios::binary);
	ydlog.precision(std::numeric_limits<cacu::float_t>::digits10);
	//log output
	std::ofstream glog(root_path + "generator_10w.log",
			ios::binary);
	glog.precision(std::numeric_limits<cacu::float_t>::digits10);


	/*
	 * create generative model
	 */
	blob *xblob_ = cacu_allocator::create_blob(batch_size, 3, img_size, img_size, train);
	blob *yblob_ = cacu_allocator::create_blob(batch_size, 3, img_size, img_size, train);

	network *xgnet = new network();
	network *xdnet = new network();

	network *ygnet = new network();
	network *ydnet = new network();

	layer_block *xgenerator = create_generator_cycleGan(batch_size, 32, train);
	xgenerator->load_weights(root_path + "models/xgenerator_100000_256.model");
	layer_block *ygenerator = create_generator_cycleGan(batch_size, 32, train);
	ygenerator->load_weights(root_path + "models/ygenerator_100000_256.model");

	/*
	 * output of the generator
	 */
	blob *xsuspicious = (blob*)xgenerator->get_oblob();
	blob *xghead = xgenerator->layers(0)->get_head_op()->in_data<blob>();
	blob *ysuspicious = (blob*)ygenerator->get_oblob();
	blob *yghead = ygenerator->layers(0)->get_head_op()->in_data<blob>();

	layer_block *xdiscriminator = create_discriminator_cycleGan(batch_size, 64, train);
	xdiscriminator->load_weights(root_path + "models/xdiscriminator_100000_256_64.model");
	layer_block *ydiscriminator = create_discriminator_cycleGan(batch_size, 64, train);
	ydiscriminator->load_weights(root_path + "models/ydiscriminator_100000_256_64.model");
	/*
	 * input of the discriminator
	 */
	blob *xdhead = xdiscriminator->layers(0)->get_head_op()->in_data<blob>();
	blob *ydhead = ydiscriminator->layers(0)->get_head_op()->in_data<blob>();

	*xgnet << xgenerator;
	*ygnet << ygenerator;
	*xdnet << xdiscriminator;
	*ydnet << ydiscriminator;

	LOG_DEBUG("Finish creating models!");

	/*
	 * create abse_op
	 */
	blobs *abse_blobsx = new blobs();
	abse_blobsx->push_back(xsuspicious);
	abse_blobsx->push_back(xblob_);

	blobs *abse_blobsy = new blobs();
	abse_blobsy->push_back(ysuspicious);
	abse_blobsy->push_back(yblob_);

	float_t lamda_ = 10;
	float_t idt_lamda_ = 0.5;
	float_t lr = 0.0002;

	abse_loss_op *abse_opx = new abse_loss_op(abse_blobsx);

	abse_loss_op *abse_opy = new abse_loss_op(abse_blobsy);
	abse_opy->set_loss_weight(lamda_);


	adam_solver *xgsgd = new adam_solver(xgnet);
	xgsgd->set_lr(lr);
	xgsgd->set_beta1(0.5);
	xgsgd->set_weight_decay(0.01);
	adam_solver *ygsgd = new adam_solver(ygnet);
	ygsgd->set_lr(lr);
	ygsgd->set_beta1(0.5);
	ygsgd->set_weight_decay(0.01);

	adam_solver *xdsgd = new adam_solver(xdnet);
	xdsgd->set_lr(lr*1);
	xdsgd->set_beta1(0.5);
	xdsgd->set_weight_decay(0.01);
	adam_solver *ydsgd = new adam_solver(ydnet);
	ydsgd->set_lr(lr*1);
	ydsgd->set_beta1(0.5);
	ydsgd->set_weight_decay(0.01);


	string xdatapath = root_path + "trainA/";
	string ydatapath = root_path + "trainB/";
	string xtrainlist = root_path + "trainA.txt";
	string ytrainlist = root_path + "trainB.txt";

	vector<string> xfull_data;
	vector<string> yfull_data;

	vector<string> vec;
	vec_i label(batch_size);

	/**
	 * read train list data into local memory
	 */
	ifstream is = ifstream(xtrainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", xtrainlist.c_str());
	string file_ = "";
	while (getline(is, file_)) {
		xfull_data.push_back(file_);
	}
	is.close();

	//random_shuffle(xfull_data.begin(), xfull_data.end());

	/**
	 * read train list data into local memory
	 */
	is = ifstream(ytrainlist);
	is.precision(numeric_limits<float>::digits10);
	if (!is)
		LOG_FATAL("file %s cannot be opened!", ytrainlist.c_str());
	while (getline(is, file_)) {
		yfull_data.push_back(file_);
	}
	is.close();

	//random_shuffle(yfull_data.begin(), yfull_data.end());

	int X_ALL_DATA_SIZE = xfull_data.size();
	int Y_ALL_DATA_SIZE = yfull_data.size();

	int step_index_x = 0;
	int step_index_y = 0;
	int step_index_ccx = 0;//X_ALL_DATA_SIZE / 2;
	int step_index_ccy = 0;//Y_ALL_DATA_SIZE / 2;
	int step_index_idtx = 0;//X_ALL_DATA_SIZE / 3;
	int step_index_idty = 0;//Y_ALL_DATA_SIZE / 3;
	time_utils *timer = new time_utils();

	unsigned int max_index;
	cacu::float_t count = 0;

	int previous_count=0;
	int allcount = 0;
	cacu::float_t xdlossf = 0,xdlosst = 0,xgloss = 0;
	cacu::float_t ydlossf = 0,ydlosst = 0,ygloss = 0;
	cacu::float_t ccloss;

	for (int i = 1; i <= max_iter; ++i) {

		timer->start();

		/*
		 * read original img data
		 */

		if (step_index_x == X_ALL_DATA_SIZE) {
			step_index_x = 0;
			random_shuffle(xfull_data.begin(), xfull_data.end());
		}
		if (step_index_y == Y_ALL_DATA_SIZE) {
			step_index_y = 0;
			random_shuffle(xfull_data.begin(), xfull_data.end());
		}
		if (step_index_ccx == X_ALL_DATA_SIZE)
			step_index_ccx = 0;
		if (step_index_ccy == Y_ALL_DATA_SIZE)
			step_index_ccy = 0;
		if (step_index_idtx == X_ALL_DATA_SIZE)
			step_index_idtx = 0;
		if (step_index_idty == Y_ALL_DATA_SIZE)
			step_index_idty = 0;

		//readimg(xblob_->s_data(),(xdatapath + xfull_data[step_index_x]).c_str());
		//readimg(yblob_->s_data(),(ydatapath + yfull_data[step_index_y]).c_str());

		/*
		 * adversrial training process
		 */
		//GAN x
		//xdloss_t
		readimg(xdhead->s_data(),(xdatapath + xfull_data[step_index_x]).c_str());
		xdiscriminator->top_op<softmax_with_loss_op>()->in_datas()->asbinblob(1)->set_data(1);
		//xdiscriminator->top_op<softmax_with_loss_op>()->set_loss_weight(0.5);
		xdnet->forward_propagate();
		xdnet->backward_propagate();
		//xdsgd->train_iter(i);
		xdlosst = xdiscriminator->top_op<softmax_with_loss_op>()->loss();

		//xdloss_f
		readimg(xghead->s_data(),(ydatapath + yfull_data[step_index_y]).c_str());
		xgnet->forward_propagate();

		xdhead->copy_blob(xsuspicious);
		xdiscriminator->top_op<softmax_with_loss_op>()->in_datas()->asbinblob(1)->set_data(0);
		xdnet->forward_propagate();
		xdnet->backward_propagate();
		xdlossf = xdiscriminator->top_op<softmax_with_loss_op>()->loss();

		xdsgd->updates(i);

		xgloss = xdiscriminator->top_op<softmax_with_loss_op>()->loss();
		cacu_copy(xdhead->s_diff(), xsuspicious->count(), xsuspicious->s_diff());
		xgnet->backward_propagate();
		//xdnet->_RESET_WEIGHT_DIFF();
		xgsgd->update_direction(maximize);

		xgsgd->updates(i);

		//GAN y
		//ydloss_t
		readimg(ydhead->s_data(),(ydatapath + yfull_data[step_index_y]).c_str());
		ydiscriminator->top_op<softmax_with_loss_op>()->in_datas()->asbinblob(1)->set_data(1);
		//ydiscriminator->top_op<softmax_with_loss_op>()->set_loss_weight(0.5);
		ydnet->forward_propagate();
		ydnet->backward_propagate();
		//ydsgd->train_iter(i);
		ydlosst = ydiscriminator->top_op<softmax_with_loss_op>()->loss();

		//ydloss_f
		readimg(yghead->s_data(),(xdatapath + xfull_data[step_index_x]).c_str());
		ygnet->forward_propagate();

		ydhead->copy_blob(ysuspicious);
		ydiscriminator->top_op<softmax_with_loss_op>()->in_datas()->asbinblob(1)->set_data(0);
		ydnet->forward_propagate();
		ydnet->backward_propagate();

		ydlossf = ydiscriminator->top_op<softmax_with_loss_op>()->loss();
		ydsgd->updates(i);

		//ygloss
		ygloss = ydiscriminator->top_op<softmax_with_loss_op>()->loss();
		cacu_copy(ydhead->s_diff(), ysuspicious->count(), ysuspicious->s_diff());
		ygnet->backward_propagate();
		ygsgd->update_direction(maximize);

		//ydnet->_RESET_WEIGHT_DIFF();
		ygsgd->updates(i);


		/*
		 * ccloss training process
		 */
		xgsgd->update_direction(minimize);
		ygsgd->update_direction(minimize);

		//F(G(y)) -> y
		abse_opy->set_loss_weight(lamda_);
		readimg(xghead->s_data(),(ydatapath + yfull_data[step_index_ccy]).c_str());
		readimg(yblob_->s_data(),(ydatapath + yfull_data[step_index_ccy]).c_str());
		xgnet->forward_propagate();

		yghead->copy_blob(xsuspicious);
		ygnet->forward_propagate();

		abse_opy->infer();
		ccloss = abse_opy->loss();
		abse_opy->derivative();

		ygnet->backward_propagate();
		cacu_copy(yghead->s_diff(),xsuspicious->count(),xsuspicious->s_diff());
		xgnet->backward_propagate();

//		ygsgd->updates(i);
//		xgsgd->updates(i);


		//G(F(x)) -> x
		abse_opx->set_loss_weight(lamda_);
		readimg(yghead->s_data(),(xdatapath + xfull_data[step_index_ccx]).c_str());
		readimg(xblob_->s_data(),(xdatapath + xfull_data[step_index_ccx]).c_str());
		ygnet->forward_propagate();

		xghead->copy_blob(ysuspicious);
		xgnet->forward_propagate();

		abse_opx->infer();
		ccloss += abse_opx->loss();
		abse_opx->derivative();

		xgnet->backward_propagate();
		cacu_copy(xghead->s_diff(),ysuspicious->count(),ysuspicious->s_diff());
		ygnet->backward_propagate();

//		xgsgd->updates(i);
//		ygsgd->updates(i);


		//indentfy constant
		//y generator
		//xgsgd->update_direction(maximize);
		//ygsgd->update_direction(maximize);
		abse_opy->set_loss_weight(lamda_ * idt_lamda_);
		readimg(yghead->s_data(),(ydatapath + yfull_data[step_index_idty]).c_str());
		readimg(yblob_->s_data(),(ydatapath + yfull_data[step_index_idty]).c_str());
		ygnet->forward_propagate();

		abse_opy->infer();
		ccloss += abse_opy->loss();
		abse_opy->derivative();
		ygnet->backward_propagate();
//		ygsgd->updates(i);

		//x generator
		abse_opx->set_loss_weight(lamda_ * idt_lamda_);
		readimg(xghead->s_data(),(xdatapath + xfull_data[step_index_idtx]).c_str());
		readimg(xblob_->s_data(),(xdatapath + xfull_data[step_index_idtx]).c_str());
		xgnet->forward_propagate();

		abse_opx->infer();
		ccloss += abse_opx->loss();
		abse_opx->derivative();
		xgnet->backward_propagate();


		//finally update the weight
		xgsgd->updates(i);
		ygsgd->updates(i);


		xdlog << xdlosst << "," << xdlossf << endl;
		xdlog.flush();

		ydlog << ydlosst << "," << ydlossf << endl;
		ydlog.flush();

		glog << xgloss << "," << ygloss << "," << ccloss<< endl;

		timer->end();

		if (i % 1 == 0) {

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, xdsgd->lr(), timer->get_time_span() / 1000);
			LOG_INFO("xdiscriminator loss : (t: %f , f: %f)", xdlosst, xdlossf);
			LOG_INFO("ydiscriminator loss : (t: %f , f: %f)", ydlosst, ydlossf);
			LOG_INFO("generator loss : (x: %f, y: %f, cc: %f)", xgloss, ygloss, ccloss);
		}

		if (i % 60000 == 0){
			xdsgd->set_lr_iter(0.1);
			xgsgd->set_lr_iter(0.1);
			ydsgd->set_lr_iter(0.1);
			ygsgd->set_lr_iter(0.1);
		}

		if (i % 1000 == 0) {
			ostringstream oss;

			yghead->copy_blob(xblob_);
			ygnet->forward_propagate();

			xghead->copy_blob(yblob_);
			xgnet->forward_propagate();

			oss << root_path << "generative/" << "x_" << i << "_" << img_size << ".jpg";
			imageio_utils::imwrite(yblob_,oss.str().c_str());
			oss.str("");

			oss << root_path << "generative/" << "x_" << i << "_g" << img_size << ".jpg";
			imageio_utils::imwrite(((blob*)xgenerator->get_oblob()),oss.str().c_str());
			oss.str("");

			oss << root_path << "generative/" << "y_" << i << "_" << img_size << ".jpg";
			imageio_utils::imwrite(xblob_,oss.str().c_str());
			oss.str("");

			oss << root_path << "generative/" << "y_" << i << "_g" << img_size << ".jpg";
			imageio_utils::imwrite(((blob*)ygenerator->get_oblob()),oss.str().c_str());
		}
		if (i % 10000 == 0) {
			ostringstream oss;

			oss << root_path << "models/" << "xgenerator_" << i << "_" << img_size << ".model";
			xgenerator->save_weights(oss.str());
			oss.str("");

			oss << root_path << "models/" << "ygenerator_" << i << "_" << img_size << ".model";
			ygenerator->save_weights(oss.str());
			oss.str("");

			oss << root_path << "models/" << "xdiscriminator_" << i << "_" << img_size << "_64.model";
			xdiscriminator->save_weights(oss.str());
			oss.str("");

			oss << root_path << "models/" << "ydiscriminator_" << i << "_" << img_size << "_64.model";
			ydiscriminator->save_weights(oss.str());

		}

		step_index_x += 1;
		step_index_y += 1;
		step_index_ccx += 1;
		step_index_ccy += 1;
		step_index_idtx += 1;
		step_index_idty += 1;
	}

	xdlog.close();
	ydlog.close();
	glog.close();

	LOG_INFO("optimization is done!");

	vector<string>().swap(xfull_data);
	vector<string>().swap(yfull_data);

	delete xdiscriminator,ydiscriminator,xgenerator,ygenerator;
	delete xgnet, ygnet;
	delete xgsgd, ygsgd;
	delete xdnet, ydnet;
	delete xdsgd, ydsgd;
	delete abse_opx,abse_opy;
	delete abse_blobsx, abse_blobsy;
	delete xblob_;
	delete yblob_;
	delete timer;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif
