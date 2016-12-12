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

#pragma once

#include "node_ops.hpp"
#include "parallel_ops.hpp"

#include "../ops/operators.h"

//node maintains every op's fuctional method and blob output data.



/*
single node for running, parallel to divided by the num of blob
*/
namespace mycnn{

	class node_base{

	public:

		node_base(){}

		node_base(operator_base *&op){

			for (unsigned int i = 0; i < op->in_datas()->size(); ++i)
			{
				edge_base *edge_ = new edge_base((*op->in_datas())[i]);
				s_edges.push_back(edge_);
			}

			edge_base *edge_ = new edge_base(op->out_data());
			o_edge = edge_;

			_op = op;
		}

		~node_base(){
			for (unsigned int i = 0; i < s_edges.size(); ++i){
				delete s_edges[i];
			}
			delete o_edge;
			delete _op;
		}

		void gather(){
			
		}

		void apply(){
			
		}

		void scatter(){
		
		}

		inline void add_sedge(edge_base *&eb){ s_edges.push_back(eb); }

		inline void set_oedge(edge_base *&eb){ o_edge = eb; }

		inline edge_base *&s_edge(int i){ return s_edges[i]; }

	protected:

		vector<edge_base*> s_edges;

		edge_base* o_edge;

		operator_base* _op;

	private:

	};


};