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

#include "../gframework/cacu_link_graphic.h"
#include "../gframework/cacu_node_graphic.h"

#include "../../ops/operator_base.h"
#include "../data_args.h"
#include "../op_args.h"

#include "cacu_graphic.h"
#include "../graphic_definition.h"
#include "../../factory/macro_factory.h"
#include "../../factory/operator_factory.h"

using namespace std;

namespace cacu{

network* phrase_network(const chars_t &modelpath, blobs *&input_data){

	vector<graphic_edge> edges;

	network* net = new network(input_data);

	std::ifstream is(modelpath, ios::binary);
	is.precision(std::numeric_limits<int>::digits10);
	if (!is)
		LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
	int edge_size = 0;
	is.read(reinterpret_cast<char*>(&edge_size), sizeof(int));
	for(int i = 0; i < edge_size; ++i){
		graphic_edge edge(0, 0, 0, 0);
		edge.load(is);
		edges.push_back(edge);
	}

	int op_count = 0;
	is.read(reinterpret_cast<char*>(&op_count), sizeof(int));
	int op_type = 0;
	is.read(reinterpret_cast<char*>(&op_type), sizeof(int));
	data_args *data_args_ = NULL;
	op_args *op_args_ = NULL;
	int _null_set = 0;
	is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
	if(_null_set){
		data_args_ = new data_args();
		data_args_->load(is);
	}
	is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
	if(_null_set){
		op_args_ = new op_args();
		op_args_->load(is);
	}
	operator_base *op = operator_factory::create_op(macro_factory::get_optype_by_index(op_type),
			input_data, data_args_, op_args_);
	if(op_type == CACU_P_SUM_ELEMWISE)
		((pruneable_sum_elemwise_op *)op)->set_load_param(true);

	*net << op;

	for(int i = 1; i < op_count; ++i){
		blobs *in_blobs = new blobs();
		for(int j = 0; j < i; ++j)
		{
			if(edges[j].to_nodeId == i){
				if(edges[j].to_blobId != in_blobs->size())
					LOG_WARNING("to blobID is %d vs %d", edges[j].to_blobId, (int)in_blobs->size());
				in_blobs->push_back(net->get_op(edges[j].from_nodeId)->out_datas()->at(edges[j].from_blobId));
			}
		}
		is.read(reinterpret_cast<char*>(&op_type), sizeof(int));
		if(in_blobs->size() == 0)
			in_blobs->push_back(net->output_blob());
		is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
		if(_null_set){
			data_args_ = new data_args();
			data_args_->load(is);
		}
		is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
		if(_null_set){
			op_args_ = new op_args();
			op_args_->load(is);
		}

		op = operator_factory::create_op(macro_factory::get_optype_by_index(op_type),
				in_blobs, data_args_, op_args_);
		if(op_type == CACU_P_SUM_ELEMWISE)
			((pruneable_sum_elemwise_op *)op)->set_load_param(true);

		*net << op;
	}
	is.close();
	return net;
}


network* phrase_network_for_pruning(const chars_t &modelpath, blobs *&input_data){

	vector<graphic_edge> edges;

	network* net = new network(input_data);

	std::ifstream is(modelpath, ios::binary);
	is.precision(std::numeric_limits<int>::digits10);
	if (!is)
		LOG_FATAL("model file %s cannot be opened!", modelpath.c_str());
	int edge_size = 0;
	is.read(reinterpret_cast<char*>(&edge_size), sizeof(int));
	for(int i = 0; i < edge_size; ++i){
		graphic_edge edge(0, 0, 0, 0);
		edge.load(is);
		edges.push_back(edge);
	}

	int op_count = 0;
	is.read(reinterpret_cast<char*>(&op_count), sizeof(int));
	int op_type = 0;
	is.read(reinterpret_cast<char*>(&op_type), sizeof(int));
	data_args *data_args_ = NULL;
	op_args *op_args_ = NULL;
	int _null_set = 0;
	is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
	if(_null_set){
		data_args_ = new data_args();
		data_args_->load(is);
	}
	is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
	if(_null_set){
		op_args_ = new op_args();
		op_args_->load(is);
	}
	if(op_type == CACU_SUM_ELEMWISE){
		op_type = CACU_P_SUM_ELEMWISE;
	}
	operator_base *op = operator_factory::create_op(macro_factory::get_optype_by_index(op_type),
			input_data, data_args_, op_args_);
	*net << op;
	for(int i = 1; i < op_count; ++i){
		blobs *in_blobs = new blobs();
		for(int j = 0; j < i; ++j){
			if(edges[j].to_nodeId == i){
				if(edges[j].to_blobId != in_blobs->size())
					LOG_WARNING("to blobID is %d vs %d", edges[j].to_blobId, (int)in_blobs->size());
				in_blobs->push_back(net->get_op(edges[j].from_nodeId)->out_datas()->at(edges[j].from_blobId));
			}
		}
		is.read(reinterpret_cast<char*>(&op_type), sizeof(int));
		if(in_blobs->size() == 0)
			in_blobs->push_back(net->output_blob());
		is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
		if(_null_set){
			data_args_ = new data_args();
			data_args_->load(is);
		}
		is.read(reinterpret_cast<char*>(&_null_set), sizeof(int));
		if(_null_set){
			op_args_ = new op_args();
			op_args_->load(is);
		}
		if(op_type == CACU_SUM_ELEMWISE){
			op_type = CACU_P_SUM_ELEMWISE;
		}

		op = operator_factory::create_op(macro_factory::get_optype_by_index(op_type),
				in_blobs, data_args_, op_args_);
		*net << op;
	}
	is.close();
	return net;
}


}

