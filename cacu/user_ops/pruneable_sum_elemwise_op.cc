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

#include "pruneable_sum_elemwise_op.h"
#include "../math/math.h"
#include "../../tools/vec_utils.h"

namespace cacu {

	pruneable_sum_elemwise_op::pruneable_sum_elemwise_op(blobs *&data, op_args *& _o_args) :
		operator_base(data, _o_args, CACU_P_SUM_ELEMWISE) {
		_INIT_OP();
		_pruned_channels = new vector<vec_i>();
		_pruned_channel = new vec_i();
	}

	pruneable_sum_elemwise_op::~pruneable_sum_elemwise_op() {
		delete _pruned_channels;
		delete _pruned_channel;
	}

	void pruneable_sum_elemwise_op::initial() {
		_out_channels = (int)_o_args->at(0);
		_original_channels = (int)_o_args->at(1);
		if (o_blobs == NULL) {
#if __USEMBEDDING__ == ON
			o_blobs = create_em_oblobs();
			o_blobs->push_back(create_em_oblob(s_blobs->at(0)->num(),
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(),
					s_blobs->at(0)->height(), _phase));
#else
			o_blobs = create_oblobs();

			o_blobs->push_back(create_oblob(s_blobs->at(0)->num(), _out_channels,
					s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
#endif
		} else {
			o_blobs->at(0)->resize(s_blobs->at(0)->num(), _out_channels,
						s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}
	}

	void pruneable_sum_elemwise_op::init_weights() {
		return;
	}

	void pruneable_sum_elemwise_op::check() {

		if(_o_args == NULL)
			LOG_FATAL("pruneable sum elemwise op args cannot equal to NULL!");
		CHECK_GE_OP((int)s_blobs->size(), 2, "blobs size must >= 2 vs %d",
				(int)s_blobs->size());
	}

	void pruneable_sum_elemwise_op::op(blobs *&s_blobs_, blobs *&o_blobs_) {

		CHECK_GE_OP((int)_pruned_channels->size(), 0, "Pruneable sum elemwise op size of pruned_channels must larger than 0 vs %d!",
				(int)_pruned_channels->size());

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);

		for (int i = 0; i < o_blob_->num(); ++i) {
			for (unsigned int j = 0; j < (s_blobs)->size(); ++j) {
				em_blob *s_blob_ = (em_blob *) s_blobs->at(j);
				cacu_saxpy(s_blob_->p_data_d(i), (float_t) 1,
						o_blob_->p_data_d(i), o_blob_->length());
			}
			o_blob_->_sync(i);
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);
		int ci = 0, o_ci = 0;
		int all_channel = _original_channels;
		int _n_length = o_blob_->channel_length() * o_blob_->num();

		for (unsigned int j = 0; j < s_blobs_->size(); ++j) {
			blob *s_blob_ = s_blobs_->asblob(j);
			s_blob_->switch_channel_data();
			ci = 0;
			o_ci = 0;
			for(int c = 0; c < all_channel; ++c){
				if(!FIND_FROM_VEC(*_pruned_channel, c)){
					if(!FIND_FROM_VEC(_pruned_channels->at(j), c)){
						cacu_saxpy(s_blob_->s_data() + ci * _n_length, (float_t)1, o_blob_->s_data() + o_ci * _n_length, _n_length);
						ci += 1;
					}
					o_ci += 1;
				}
			}
		}
		o_blob_->switch_body();
		o_blob_->switch_channel();
#endif
	}

	void pruneable_sum_elemwise_op::grad(blobs *&s_blobs_,blobs *&o_blobs_) {

#if __USEMBEDDING__ == ON
		em_blob *o_blob_ = (em_blob*) o_blobs->at(0);

		for (unsigned int j = 0; j < (s_blobs)->size(); ++j) {
			em_blob *s_blob_ = (em_blob *) s_blobs->at(j);
			cacu_copy_cpu(o_blob_->s_diff(), o_blob_->count(),
					s_blob_->s_diff());
		}
#else
		blob *o_blob_ = o_blobs_->asblob(0);

		int ci = 0, o_ci = 0;
		int all_channel = _original_channels;
		int _n_length = o_blob_->channel_length() * o_blob_->num();

		o_blob_->switch_channel();

		for (unsigned int j = 0; j < (s_blobs_)->size(); ++j) {
			blob *s_blob_ = s_blobs_->asblob(j);
			ci = 0;
			o_ci = 0;
			for(int c = 0; c < all_channel; ++c){
				if(!FIND_FROM_VEC(*_pruned_channel, c)){
					if(!FIND_FROM_VEC(_pruned_channels->at(j), c)){
						cacu_copy(o_blob_->s_diff() + o_ci * _n_length, _n_length, s_blob_->s_diff() + ci * _n_length);
						ci += 1;
					}
					o_ci += 1;
				}
			}
			s_blob_->switch_body();
			s_blob_->switch_channel();
		}
		o_blob_->switch_body();
#endif
	}

	void pruneable_sum_elemwise_op::load(std::ifstream& is) {

		if(_load_param){
			int size_;
			is.read(reinterpret_cast<char*>(&size_), sizeof(int));
			int p_size_;
			for(int i = 0; i < size_; ++i){
				vec_i pruned_channel_;
				is.read(reinterpret_cast<char*>(&p_size_), sizeof(int));
				for(int j = 0; j < p_size_; ++j){
					int data_;
					is.read(reinterpret_cast<char*>(&data_), sizeof(int));
					pruned_channel_.push_back(data_);
				}
				_pruned_channels->push_back(pruned_channel_);
			}
			is.read(reinterpret_cast<char*>(&p_size_), sizeof(int));
			for(int j = 0; j < p_size_; ++j){
				int data_;
				is.read(reinterpret_cast<char*>(&data_), sizeof(int));
				_pruned_channel->push_back(data_);
			}
		}
	}

	void pruneable_sum_elemwise_op::save(std::ostream& os) {

		int size_ = _pruned_channels->size();
		os.write((char*)(&size_), sizeof(int));
		for(int i = 0; i < _pruned_channels->size(); ++i){
			size_ = _pruned_channels->at(i).size();
			os.write((char*)(&size_), sizeof(int));
			for(int j = 0; j < _pruned_channels->at(i).size(); ++j){
				size_ = _pruned_channels->at(i)[j];
				os.write((char*)(&(size_)), sizeof(int));
			}
		}
		size_ = _pruned_channel->size();
		os.write((char*)(&size_), sizeof(int));
		for(int j = 0; j < _pruned_channel->size(); ++j){
			size_ = _pruned_channel->at(j);
			os.write((char*)(&(size_)), sizeof(int));
		}
	}

	void pruneable_sum_elemwise_op::echo() {
		LOG_INFO("pruned sum elemwise op: in(%d, %d, %d), out(%d, %d, %d), origial_channel(%f)",
				s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
				o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height(),
				_o_args->at(1));
	}

	void pruneable_sum_elemwise_op::prune_blobs(){
		CHECK_EQ_OP((int)s_blobs->size(), (int)_pruned_channels->size(), "blobs size must == pruned_channels %d vs %d",
				(int)s_blobs->size(), (int)_pruned_channels->size());
		bool over_lap = true;
		//max overlapping
		for(int n = 0 ; n < _pruned_channels->at(0).size(); ++n){
			for(int i = 1 ; i < _pruned_channels->size(); ++i) {
				if(!FIND_FROM_VEC(_pruned_channels->at(i), _pruned_channels->at(0)[n])){
					over_lap = false;
					break;
				}
			}
			if(over_lap)
				_pruned_channel->push_back(_pruned_channels->at(0)[n]);
		}
		_out_channels = _original_channels - _pruned_channel->size();
	}

REGISTER_OP_CLASS_D_OP(pruneable_sum_elemwise_op);
REGISTER_OP_MACRO(pruneable_sum_elemwise_op, CACU_P_SUM_ELEMWISE);

}
