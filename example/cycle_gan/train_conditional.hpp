
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

#ifndef TRAIN_CONDITIONAL_CYCLE_GAN_HPP_
#define TRAIN_CONDITIONAL_CYCLE_GAN_HPP_

#include "../../cacu/cacu.h"

#include "../../cacu/solvers/solvers.h"
//#include "../../cacu/solvers/sgd_solver.h"

#include "../../tools/string_utils.h"
#include "../frcnn/tools_frcnn.h"

#include "../imagenet/data_proc.h"
#include "conditional_models.h"

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
	std::ofstream glog(root_path + "generator.log",
			ios::binary);
	glog.precision(std::numeric_limits<cacu::float_t>::digits10);


	/*
	 * create generative model
	 */
	blob *inblob_ = cacu_allocator::create_blob(batch_size, 3, img_size, img_size, train);

	blob *temp = cacu_allocator::create_blob(batch_size, 3, img_size, img_size, test);
	bin_blob *conditional_label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1, test);
	//blob *yblob_ = cacu_allocator::create_blob(batch_size, 3, img_size, img_size, train);

	network *gnet = new network();
	network *dnet = new network();

	layer_block *generator = create_generator_cycleGan(batch_size, conditional_label_, 64, train);
	//xgenerator->load_weights(root_path + "models/xgenerator_50000_256.model");

	/*
	 * output of the generator
	 */
	blob *suspicious = (blob*)generator->get_oblob();
	blob *ghead = generator->layers(0)->get_head_op()->in_data<blob>();

	layer_block *discriminator = create_discriminator_cycleGan(batch_size, conditional_label_, 64, train);
	//xdiscriminator->load_weights(root_path + "models/xdiscriminator_50000_256.model");

	/*
	 * input of the discriminator
	 */
	blob *dhead = discriminator->layers(0)->get_head_op()->in_data<blob>();

	*gnet << generator;
	*dnet << discriminator;

	LOG_DEBUG("Finish creating models!");

	/*
	 * create abse_op
	 */
	blobs *abse_blobs = new blobs();
	abse_blobs->push_back(suspicious);
	abse_blobs->push_back(inblob_);

	float_t lamda_ = 1;
	float_t idt_lamda_ = 0.5;
	float_t lr = 0.0002;

	abse_loss_op *abse_op = new abse_loss_op(abse_blobs);
	abse_op->set_loss_weight(lamda_);

	adam_solver *gsgd = new adam_solver(gnet);
	gsgd->set_lr(lr);
	gsgd->set_beta1(0.5);
	gsgd->set_weight_decay(0.01);

	adam_solver *dsgd = new adam_solver(dnet);
	dsgd->set_lr(lr);
	dsgd->set_beta1(0.5);
	dsgd->set_weight_decay(0.01);

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
	ifstream is;
	is.open(xtrainlist);
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
	is.open(ytrainlist);
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
	cacu::float_t ccloss = 0;

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

		/*
		 * adversrial training process
		 */
		//GAN x
		//xdloss_t
		//projection setting
		conditional_label_->set_data(0);

		readimg(dhead->s_data(),(xdatapath + xfull_data[step_index_x]).c_str());
		discriminator->top_op<sigmoid_with_loss_op>()->in_datas()->asbinblob(1)->set_data(1);
		//xdiscriminator->top_op<softmax_with_loss_op>()->set_loss_weight(0.5);
		dnet->forward_propagate();
		dnet->backward_propagate();
		//xdsgd->train_iter(i);
		xdlosst = discriminator->top_op<sigmoid_with_loss_op>()->loss();


		//xdloss_f
		readimg(ghead->s_data(),(ydatapath + yfull_data[step_index_y]).c_str());
		gnet->forward_propagate();

		dhead->copy_blob(suspicious);
		discriminator->top_op<sigmoid_with_loss_op>()->in_datas()->asbinblob(1)->set_data(0);
		dnet->forward_propagate();
		dnet->backward_propagate();
		xdlossf = discriminator->top_op<sigmoid_with_loss_op>()->loss();

		dsgd->updates(i);

		xgloss = discriminator->top_op<sigmoid_with_loss_op>()->loss();
		cacu_copy(dhead->s_diff(), suspicious->count(), suspicious->s_diff());
		gnet->backward_propagate();
		//xdnet->_RESET_WEIGHT_DIFF();
		gsgd->update_direction(maximize);

		gsgd->updates(i);

		//GAN y
		//ydloss_t
		//projection setting
		conditional_label_->set_data(1);
		readimg(dhead->s_data(),(ydatapath + yfull_data[step_index_y]).c_str());
		discriminator->top_op<sigmoid_with_loss_op>()->in_datas()->asbinblob(1)->set_data(1);
		//ydiscriminator->top_op<softmax_with_loss_op>()->set_loss_weight(0.5);
		dnet->forward_propagate();
		dnet->backward_propagate();
		//ydsgd->train_iter(i);
		ydlosst = discriminator->top_op<sigmoid_with_loss_op>()->loss();

		//ydloss_f
		readimg(ghead->s_data(),(xdatapath + xfull_data[step_index_x]).c_str());
		gnet->forward_propagate();

		dhead->copy_blob(suspicious);
		discriminator->top_op<sigmoid_with_loss_op>()->in_datas()->asbinblob(1)->set_data(0);
		dnet->forward_propagate();
		dnet->backward_propagate();

		ydlossf = discriminator->top_op<sigmoid_with_loss_op>()->loss();
		dsgd->updates(i);

		//ygloss
		ygloss = discriminator->top_op<sigmoid_with_loss_op>()->loss();
		cacu_copy(dhead->s_diff(), suspicious->count(), suspicious->s_diff());
		gnet->backward_propagate();
		gsgd->update_direction(maximize);

		//ydnet->_RESET_WEIGHT_DIFF();
		gsgd->updates(i);


		/*
		 * ccloss training process
		 */

		gsgd->update_direction(minimize);

		//F(G(y)) -> y
		conditional_label_->set_data(0);
		abse_op->set_loss_weight(lamda_);
		readimg(ghead->s_data(),(ydatapath + yfull_data[step_index_ccy]).c_str());
		readimg(inblob_->s_data(),(ydatapath + yfull_data[step_index_ccy]).c_str());
		gnet->forward_propagate();

		ghead->copy_blob(suspicious);
		conditional_label_->set_data(1);
		gnet->forward_propagate();

		abse_op->infer();
		ccloss = abse_op->loss();
		abse_op->derivative();

		gnet->backward_propagate();
		cacu_copy(ghead->s_diff(),temp->count(),temp->s_data());
		conditional_label_->set_data(0);
		readimg(ghead->s_data(),(ydatapath + yfull_data[step_index_ccy]).c_str());
		gnet->forward_propagate();
		cacu_copy(temp->s_data(),suspicious->count(),suspicious->s_diff());
		gnet->backward_propagate();

//		ygsgd->updates(i);
//		xgsgd->updates(i);


		//G(F(x)) -> x
		conditional_label_->set_data(1);
		abse_op->set_loss_weight(lamda_);
		readimg(ghead->s_data(),(xdatapath + xfull_data[step_index_ccx]).c_str());
		readimg(inblob_->s_data(),(xdatapath + xfull_data[step_index_ccx]).c_str());
		gnet->forward_propagate();

		ghead->copy_blob(suspicious);
		conditional_label_->set_data(0);
		gnet->forward_propagate();

		abse_op->infer();
		ccloss += abse_op->loss();
		abse_op->derivative();

		gnet->backward_propagate();
		cacu_copy(ghead->s_diff(),temp->count(),temp->s_data());
		conditional_label_->set_data(1);
		readimg(ghead->s_data(),(xdatapath + xfull_data[step_index_ccx]).c_str());
		gnet->forward_propagate();
		cacu_copy(temp->s_data(),suspicious->count(),suspicious->s_diff());
		gnet->backward_propagate();
		//*/
//		xgsgd->updates(i);
//		ygsgd->updates(i);


		//indentfy constant
		//y generator
		//xgsgd->update_direction(maximize);
		//ygsgd->update_direction(maximize);
		conditional_label_->set_data(1);
		abse_op->set_loss_weight(lamda_ * idt_lamda_);
		readimg(ghead->s_data(),(ydatapath + yfull_data[step_index_idty]).c_str());
		readimg(inblob_->s_data(),(ydatapath + yfull_data[step_index_idty]).c_str());
		gnet->forward_propagate();

		abse_op->infer();
		ccloss += abse_op->loss();
		abse_op->derivative();
		gnet->backward_propagate();
//		ygsgd->updates(i);

		//x generator
		conditional_label_->set_data(0);
		abse_op->set_loss_weight(lamda_ * idt_lamda_);
		readimg(ghead->s_data(),(xdatapath + xfull_data[step_index_idtx]).c_str());
		readimg(inblob_->s_data(),(xdatapath + xfull_data[step_index_idtx]).c_str());
		gnet->forward_propagate();

		abse_op->infer();
		ccloss += abse_op->loss();
		abse_op->derivative();
		gnet->backward_propagate();

		//finally update the weight
		gsgd->updates(i);

		xdlog << xdlosst << "," << xdlossf << endl;
		xdlog.flush();

		ydlog << ydlosst << "," << ydlossf << endl;
		ydlog.flush();

		glog << xgloss << "," << ygloss << "," << ccloss<< endl;

		timer->end();

		if (i % 1 == 0) {

			LOG_INFO("iter_%d, lr: %f, %ld ms/iter", i, dsgd->lr(), timer->get_time_span() / 1000);
			LOG_INFO("xdiscriminator loss : (t: %f , f: %f)", xdlosst, xdlossf);
			LOG_INFO("ydiscriminator loss : (t: %f , f: %f)", ydlosst, ydlossf);
			LOG_INFO("generator loss : (x: %f, y: %f, cc: %f)", xgloss, ygloss, ccloss);
		}

		if (i % 80000 == 0){
			dsgd->set_lr_iter(0.1);
			gsgd->set_lr_iter(0.1);
		}

		if (i % 1000 == 0) {
			ostringstream oss;

			ghead->copy_blob(inblob_);
			conditional_label_->set_data(0);
			gnet->forward_propagate();

			oss << root_path << "generative/" << "x_" << i << "_" << img_size << ".jpg";
			imageio_utils::imwrite(inblob_,oss.str().c_str());
			oss.str("");

			oss << root_path << "generative/" << "x_" << i << "_g" << img_size << ".jpg";
			imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());
			oss.str("");

			conditional_label_->set_data(1);
			gnet->forward_propagate();

			oss << root_path << "generative/" << "y_" << i << "_" << img_size << ".jpg";
			imageio_utils::imwrite(inblob_,oss.str().c_str());
			oss.str("");

			oss << root_path << "generative/" << "y_" << i << "_g" << img_size << ".jpg";
			imageio_utils::imwrite(((blob*)generator->get_oblob()),oss.str().c_str());
		}
		if (i % 10000 == 0) {
			ostringstream oss;

			oss << root_path << "models/" << "xgenerator_" << i << "_" << img_size << "_64.model";
			generator->save_weights(oss.str());
			oss.str("");

			oss << root_path << "models/" << "xdiscriminator_" << i << "_" << img_size << "_64.model";
			discriminator->save_weights(oss.str());
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

	delete discriminator,generator;
	delete gnet;
	delete gsgd;
	delete dnet;
	delete dsgd;
	delete abse_op;
	delete abse_blobs;
	delete inblob_;
	delete timer;
	delete temp;

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cuda_release();
#endif
#endif
}


#endif

