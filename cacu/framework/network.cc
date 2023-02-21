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

#include "network.h"

#include "../../tools/vec_utils.h"

namespace cacu {



network::network() {
	_ops = new vector<operator_base*>();
	_layers = new vector<layer*>();
#if __OPERATOR__TYPE__ == __DYNAMIC_GRAPH__
	LOG_INFO("Create model from dynamic operators!");
#elif __OPERATOR__TYPE__ == __STATIC_GRAPH__
	LOG_INFO("Create model from static operators!");
#endif
}


network::network(blobs *&datas_) {
	_input_blobs = datas_;

	_ops = new vector<operator_base*>();
	_layers = new vector<layer*>();
#if __OPERATOR__TYPE__ == __DYNAMIC_GRAPH__
	LOG_INFO("Create model from dynamic operators!");
#elif __OPERATOR__TYPE__ == __STATIC_GRAPH__
	LOG_INFO("Create model from static operators!");
#endif
}

network::~network() {

	//network have the priority to delete input data blob
	if(_input_blobs != NULL){
		//delete _input_blobs;
		_input_blobs = NULL;
	}

	for(unsigned int i = 0; i < _layers->size(); ++i) {
		delete _layers->at(i);
		_layers->at(i) = NULL;
	}

	for(int i = 0; i < op_count(); ++i) {
		delete get_op(i);
	}
	delete _layers;
	_layers = NULL;

	delete _ops;
	_ops = NULL;
}

/*
 * Class network unlike layer & layer_block,
 * it maintains the abstract layer's operator entity.
 */
network& network::operator <<(layer_block* layer_block_) {
	for (int i = 0; i < layer_block_->length(); ++i) {
		_layers->push_back(layer_block_->layer_bases(i));
		for(int j = 0; j < layer_block_->layer_bases(i)->op_count(); ++j)
			_ops->push_back(layer_block_->layer_bases(i)->get_op_base(j));
	}
	if(layer_block_->get_indatas() != NULL)
		_input_blobs = layer_block_->get_indatas();
	//delete layer_block_;
	//layer_block_ = NULL;
	_ASSOCIATE_INPUT_DATA();
	return *this;
}

network& network::operator <<(layer_block& layer_block_) {
	for(int i = 0; i < layer_block_.length(); ++i) {
		_layers->push_back(layer_block_.layer_bases(i));
		for(int j = 0; j < layer_block_.layer_bases(i)->op_count(); ++j)
			_ops->push_back(layer_block_.layer_bases(i)->get_op_base(j));
	}
	if(layer_block_.get_indatas() != NULL)
		_input_blobs = layer_block_.get_indatas();
	//delete &layer_block_;
	_ASSOCIATE_INPUT_DATA();
	return *this;
}

network& network::operator <<(layer* const &layer_) {

	_layers->push_back(layer_);
	for(int j = 0; j < layer_->op_count(); ++j)
		_ops->push_back(layer_->get_op_base(j));
	_ASSOCIATE_INPUT_DATA();
	return *this;
}

network& network::operator <<(layer& layer_) {

	_layers->push_back(&layer_);
	for(int j = 0; j < layer_.op_count(); ++j)
		_ops->push_back(layer_.get_op_base(j));
	_ASSOCIATE_INPUT_DATA();
	return *this;
}

network& network::operator <<(operator_base*& op_){
	_ops->push_back(op_);
	op_->echo();
	_ASSOCIATE_INPUT_DATA();
	return *this;
}

void network::predict() {
	//if (test != phase()) {
	set_phase(test);
	//}
	for(int i = 0 ; i < op_count(); ++i) {
		//LOG_DEBUG("%d: %d", i, get_op(i)->get_optype());
		get_op(i)->infer();
	}
}

void network::forward_propagate(int op_num) {
	for(int i = op_num; i < op_count(); ++i) {
//		LOG_DEBUG("forward op: %d",i);
		get_op(i)->infer();
//		if(get_op(i)->get_op_name() == "dthreshold"){
//			LOG_DEBUG("%d", i);
//			cacu_print(get_op(i)->in_datas()->asblob(0)->s_data(), 100);
//			cacu_print(get_op(i)->in_datas()->asblob(1)->s_data(), 100);
//		}
	}
}

void network::forward_to(int op_num) {
	assert(op_num < op_count());
	for(int i = 0; i <= op_num; ++i) {
//		LOG_DEBUG("forward op: %d",i);
		get_op(i)->infer();
//		if(get_op(i)->get_op_name() == "Gout"){
//			LOG_DEBUG("%d", i);
//			cacu_print(get_op(i)->in_datas()->asblob(0)->s_data(), 100);
//			cacu_print(get_op(i)->in_datas()->asblob(1)->s_data(), 100);
//		}
	}
}

void network::backward_propagate(int op_num) {
	if(op_num < 0)
		op_num = op_count() - 1;
	if(this->phase() == test)
		LOG_FATAL("Network is created with a test phase. It's not a safe to call the function!");
	//et_op(0)->__NEED_BACK_PROPAGATE__(false);
	for(int i = op_num; i >= 0; --i) {
		get_op(i)->derivative();
//		LOG_DEBUG("backward op: %d",i);
	}
}

void network::backward_to(int op_num) {
	CHECK_GE_OP(op_num, 0, "back to %d?", op_num);
	if(this->phase() == test)
		LOG_FATAL("Network is created with a test phase. It's not a safe to call the function!");
	//et_op(0)->__NEED_BACK_PROPAGATE__(false);
	for(int i = op_count() - 1; i >= op_num; --i) {
		get_op(i)->derivative();
//		if(get_op(i)->get_op_name() == "checker"){
//			LOG_DEBUG("%d", i);
//			cacu_print(get_op(i)->in_datas()->asblob(0)->s_diff(), 100);
//			cacu_print(get_op(i)->in_datas()->asblob(1)->s_diff(), 100);
//		}
//		LOG_DEBUG("backward op: %d",i);
	}
}

void network::set_weights_type(param_init_type type_, float_t value) {
	for(int i = op_count() - 1; i >= 0; --i) {
		for(int j = 0; j < get_op(i)->weights_size(); ++j) {
			get_op(i)->get_weight(j)->set_init_type(type_, value);
		}
	}
}

void network::output_blobs() {
	for(int i = 0; i < op_count(); ++i) {
		blob_base * bp = get_op(i)->out_data<blob_base>();
		if(bp != NULL && bp->is_output())
			LOG_DEBUG("%d", i);
		else if(bp == NULL) {
			for(unsigned int j = 0; j < get_op(i)->out_datas()->size(); ++j) {
				if(get_op(i)->out_datas()->at(j)->is_output())
					LOG_DEBUG("%d:%d", i, j);
			}
		}
	}
}

void network::load_weights_from(std::ifstream &is, int op_start, int op_end) {

	CHECK_LE_OP(op_end,op_count(),"end index must less than current op count!");
	CHECK_LT_OP(op_start,op_end,"start index must less than end index!");
	for(int i = op_start; i < op_end; ++i) {
		get_op(i)->load(is);
	}
	LOG_INFO("Initialize model from op[%d] to op[%d]", op_start, op_end);
}

void network::check() {
	LOG_INFO("Check NetWork!");
	for(unsigned int i = 0; i < _ops->size(); ++i) {
		_ops->at(i)->echo();
	}
}

void network::set_phase(phase_type phase_) {
	for(unsigned int i = 0; i < _ops->size(); ++i) {
		_ops->at(i)->set_phase(phase_);
	}
}

void network::set_update_weight(bool isupdate_) {
	for(unsigned int i = 0; i < _ops->size(); ++i) {
		_ops->at(i)->set_is_update_weight(isupdate_);
//		if(_ops->at(i)->_TYPE() == CACU_BATCH_NORMALIZE)
//			(dynamic_cast<batch_normalize_op*>(_ops->at(i)))->set_is_use_global_stats(isupdate_==false);
	}
}

void network::upgrade_bn_initial(){
	vec_i *upgrade_index;
	for (unsigned int i = 0; i < _ops->size(); ++i) {
		if(_ops->at(i)->_TYPE() == CACU_BATCH_NORMALIZE && i > 0) {
			if(_ops->at(i - 1)->get_optype() == CACU_INNERPRODUCT || _ops->at(i - 1)->get_optype() == CACU_CONVOLUTION){
				upgrade_index = _ops->at(i - 1)->get_weight(0)->upgrade_index();
				for(int c = 0; c < _ops->at(i)->get_weight(0)->num(); ++c)
					if(!FIND_FROM_VEC(*upgrade_index, c)){
						//initial scale by 1
						_ops->at(i)->get_weight(0)->set_pdata(1,c);
						//initial shift by 0
						_ops->at(i)->get_weight(1)->set_pdata(0,c);
						//initial history by 0
						_ops->at(i)->get_storage_blob(9)->set_pdata(0, c);
						_ops->at(i)->get_storage_blob(10)->set_pdata(0, c);
					}
			}
		}
	}
}

vector<graphic_edge> network::_phrase_tensor_path(){
	vector<graphic_edge> edges;
	for(int i = 0; i < op_count(); ++i) {
		if(get_op(i)->is_alloc_output()) {
			for(unsigned int n = 0; n < get_op(i)->out_datas()->size(); ++n) {
				for(int j = 1; j < op_count(); ++j) {
					if(!get_op(j)->is_alloc_output())
						continue;
					int blobId_ = get_op(j)->in_datas()->is_contain(get_op(i)->out_datas()->at(n));
					if(blobId_ >= 0) {
						graphic_edge edge(i, n, j, blobId_);
						edges.push_back(edge);
						break;
					}
				}
			}
		}
	}
	return edges;
}

void network::serialize_model(const chars_t& modelpath) {
	std::ofstream os(modelpath, ios::binary);
	os.precision(std::numeric_limits<int>::digits10);

	vector<graphic_edge> edges = _phrase_tensor_path();
	int size_ = edges.size();
	os.write((char*)(&size_), sizeof(int));

	for(int i = 0; i < size_; ++i) {
		edges[i].serialize(os);
	}

	size_ = op_count();
	os.write((char*)(&size_), sizeof(int));
	for(int i = 0; i < op_count(); ++i) {
		if(get_op(i)->get_basetype() == LOSS_BASE)
			continue;
		else if(get_op(i)->get_optype() == CACU_SUM_ELEMWISE){
			//add op_args to prundable_sum_element_op
			get_op(i)->get_op_args() = new op_args(get_op(i)->out_datas()->at(0)->channel(), get_op(i)->out_datas()->at(0)->channel());
			get_op(i)->serialize(os);
		}
		else
			get_op(i)->serialize(os);
	}

	vector<graphic_edge>().swap(edges);
	os.close();
	LOG_INFO("The model structure is saved at : %s", modelpath.c_str());
}

void network::_ASSOCIATE_INPUT_DATA() {
	if(_input_blobs == NULL)
	{
		if(_ops->size() != 0)
		{
			_input_blobs = new blobs();
			_input_blobs->push_back(_ops->at(0)->in_data<blob_base>());
		}
	}
}

vec_i network::_weight_clip(weight *&w_){
	vec_t temp(w_->length());
	vec_t wtemp(w_->count());
	vec_i _channel_pruned;
	if(w_->length() == 1)
		return _channel_pruned;
	float wm = 0, ws = 0;
#if __USE_DEVICE__ == ON
	cacu_sumbysize_cpu(BYWIDTH, w_->s_data_cpu(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
#else
	cacu_sumbysize(BYWIDTH, w_->s_data(), w_->count(), (float_t)1, &wm, (float_t)0, w_->count());
#endif
	wm = wm / w_->count();
#if __USE_DEVICE__ == ON
	cacu_sdxsize_cpu(w_->s_data_cpu(false), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
	cacu_sqr_cpu(w_->s_data_cpu(false), w_->count(), &wtemp[0]);
#else
	cacu_sdxsize(w_->s_data(), w_->count(), (float_t)-1 * wm, (float_t)1, &wtemp[0]);
	cacu_sqr(w_->s_data(), w_->count(), &wtemp[0]);
#endif
	cacu_sumbysize_cpu(BYWIDTH, &wtemp[0], w_->count(), (float_t)1, &ws, (float_t)0, w_->count());
	ws = sqrtf(ws / (w_->count()));
	for(int i = 0; i < w_->num(); ++i){
		float m = 0, s = 0;
#if __USE_DEVICE__ == ON
		cacu_sumbysize_cpu(BYWIDTH, w_->p_data_cpu(i, false), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
#else
		cacu_sumbysize(BYWIDTH, w_->p_data(i), w_->length(), (float_t)1, &m, (float_t)0, w_->length());
#endif
		m = m / w_->length();
#if __USE_DEVICE__ == ON
		cacu_sdxsize_cpu(w_->p_data_cpu(i, false), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr_cpu(w_->p_data_cpu(i, false), w_->length(), &temp[0]);
#else
		cacu_sdxsize(w_->p_data(i), w_->length(), (float_t)-1 * m, (float_t)1, &temp[0]);
		cacu_sqr(w_->p_data(i), w_->length(), &temp[0]);
#endif
		cacu_sumbysize_cpu(BYWIDTH, &temp[0], w_->length(), (float_t)1, &s, (float_t)0, w_->length());
		s = sqrtf(s / (w_->length()));
		if(s < ws / w_->length()){
//			cacu_set_value(w_->p_data(i), 0, w_->length());
			w_->set_pdata(0, i);
			_channel_pruned.push_back(i);
		}
	}
	return _channel_pruned;
}

void network::network_pruning(const chars_t& model_path){

	blob *new_weight = new blob(1, 1, 1, 1, 0, test);
	map<int, vec_i> _channel_pruneds;
	vec_i in_op_indexs;
	int inchannel, outchannel;
	int j;
	//phase calculation graphic
	vector<graphic_edge> edges = _phrase_tensor_path();

	for(int i = 0; i < op_count(); ++i) {
		in_op_indexs.clear();
		for(int find_id = 0 ; find_id < edges.size(); ++find_id){
			if(edges[find_id].to_nodeId == i){
				in_op_indexs.push_back(edges[find_id].from_nodeId);
			}
		}
		if((get_op(i)->get_optype() == CACU_CONVOLUTION || get_op(i)->get_optype() == CACU_INNERPRODUCT)) {
			_channel_pruneds.insert(pair<int, vec_i>(i, _weight_clip(get_op(i)->get_weight(0))));
			if(in_op_indexs.size() == 0)
				inchannel = 0;
			else
				inchannel = _channel_pruneds[in_op_indexs[0]].size();
			outchannel = _channel_pruneds[i].size();
			new_weight->resize((get_op(i)->get_weight(0)->num() - outchannel),
					(get_op(i)->get_weight(0)->channel() - inchannel), get_op(i)->get_weight(0)->width(), get_op(i)->get_weight(0)->height());

			j = 0;
			for(int n = 0 ; n < get_op(i)->get_weight(0)->num(); ++n)
				for(int c = 0 ; c < get_op(i)->get_weight(0)->channel(); ++c){
					if(!FIND_FROM_VEC(_channel_pruneds[i], n)){
						if(in_op_indexs.size() > 0){
							if(!FIND_FROM_VEC(_channel_pruneds[in_op_indexs[0]], c)){
								cacu_copy(get_op(i)->get_weight(0)->p_data(n) + c * get_op(i)->get_weight(0)->channel_length(), get_op(i)->get_weight(0)->channel_length(),
										new_weight->s_data() + j * get_op(i)->get_weight(0)->channel_length());
								j += 1;
							}
						}
						else{
							cacu_copy(get_op(i)->get_weight(0)->p_data(n) + c * get_op(i)->get_weight(0)->channel_length(), get_op(i)->get_weight(0)->channel_length(),
										new_weight->s_data() + j * get_op(i)->get_weight(0)->channel_length());
								j += 1;
						}
					}
				}
			get_op(i)->get_weight(0)->resize(get_op(i)->get_weight(0)->num() - outchannel, get_op(i)->get_weight(0)->channel() - inchannel,
					get_op(i)->get_weight(0)->width(), get_op(i)->get_weight(0)->height());
			cacu_copy(new_weight->s_data(), new_weight->count(), get_op(i)->get_weight(0)->s_data());
			get_op(i)->get_weight(0)->blob_size();
		}

		else if(get_op(i)->get_optype() == CACU_BATCH_NORMALIZE){

			vec_i _channel_pruned;
			_channel_pruned.assign(_channel_pruneds[in_op_indexs[0]].begin(), _channel_pruneds[in_op_indexs[0]].end());
			_channel_pruneds.insert(pair<int, vec_i>(i, _channel_pruned));
			new_weight->resize((get_op(i)->get_weight(0)->num() - _channel_pruned.size()), 1, 1, 1);
			for(int c = 0 ; c < get_op(i)->weights_size(); ++c){
				j = 0;
				for(int n = 0 ; n < get_op(i)->get_weight(c)->num(); ++n)
					if(!FIND_FROM_VEC(_channel_pruned, n)){
						cacu_copy(get_op(i)->get_weight(c)->p_data(n), 1,
															new_weight->s_data() + j);
						j += 1;
					}
				get_op(i)->get_weight(c)->resize(new_weight->count(), 1, 1, 1);
				cacu_copy(new_weight->s_data(), new_weight->count(), get_op(i)->get_weight(c)->s_data());
			}
			//
			for(int c = 7; c < 11; ++c){
				j = 0;
				for(int n = 0 ; n < get_op(i)->get_storage_blob(c)->num(); ++n)
					if(!FIND_FROM_VEC(_channel_pruned, n)){
						cacu_copy(get_op(i)->get_storage_blob(c)->p_data(n), 1,
															new_weight->s_data() + j);
						j += 1;
					}
				get_op(i)->get_storage_blob(c)->resize(new_weight->count(), 1, 1, 1);
				cacu_copy(new_weight->s_data(), new_weight->count(), get_op(i)->get_storage_blob(c)->s_data());
			}
		}

		else if(get_op(i)->get_optype() == CACU_P_SUM_ELEMWISE){
			for(int c = 0 ; c < in_op_indexs.size(); ++c)
				get_op<pruneable_sum_elemwise_op>(i)->add_prune_channel(_channel_pruneds[in_op_indexs[c]]);
			get_op<pruneable_sum_elemwise_op>(i)->prune_blobs();
			get_op<pruneable_sum_elemwise_op>(i)->_prepare_for_pruning();
			_channel_pruneds.insert(pair<int, vec_i>(i, *get_op<pruneable_sum_elemwise_op>(i)->get_pruned_vec()));
		}
		else {
			vec_i _channel_pruned;
			_channel_pruned.assign(_channel_pruneds[in_op_indexs[0]].begin(), _channel_pruneds[in_op_indexs[0]].end());
			_channel_pruneds.insert(pair<int, vec_i>(i, _channel_pruned));
		}

		if(get_op(i)->get_data_args() != NULL) {
			get_op(i)->get_data_args()->at(0) -= _channel_pruneds[i].size();
			get_op(i)->get_data_args()->at(4) = get_op(i)->in_datas()->at(0)->channel();
		}

		get_op(i)->initial();
	}
	this->check();
	this->serialize_model(model_path);
	delete new_weight;
}


}
