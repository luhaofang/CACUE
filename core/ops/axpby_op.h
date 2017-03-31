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

namespace mycnn{

	class axpby_op : public math_operator_base
	{

	public:

		axpby_op(blobs *&data, math_args *&args_) : math_operator_base(data, args_){

			o_blob = data->at(1);

		};

		~axpby_op(){

		};

		virtual const void check() override{
			blob_base *x = s_blobs->at(0);
			blob_base *y = s_blobs->at(1);
			//blob size check
			CHECK_GT_OP(x->count(), y->count(),"x length must equal to y %d vs %d",x->count(),y->count());
		}

		virtual const void op() override {

			blob *x = (blob*)s_blobs->at(0);
			blob *y = (blob*)s_blobs->at(1);

			cacu_saxpby(x->s_data(),_args->at(0),y->s_data(),_args->at(1),x->count());

		}


		virtual const void grad() override {
			blob *x = (blob*)s_blobs->at(0);
			blob *y = (blob*)s_blobs->at(1);

			cacu_scalex(x->s_diff(),_args->at(0),x->count());
			cacu_scalex(y->s_diff(),_args->at(1),y->count());
		}


	private:


	};
};
