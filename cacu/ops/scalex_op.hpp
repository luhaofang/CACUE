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

#ifndef SCALEX_OP_HPP_
#define SCALEX_OP_HPP_

namespace cacu {

	class scalex_op : public math_op {

	public:

		scalex_op(blobs *&data, op_args *&args_) :
			math_op(data, args_, CACU_MATH_SCALEX) {
		}

		~scalex_op() {
		}

		void initial() override {
			if (o_blobs == NULL){
				o_blobs = s_blobs;
			}
			else
				o_blobs->_NEED_MOTIFY();
		}

		void check() override {
			if(_o_args == NULL)
				LOG_FATAL("scalex op args cannot equal to NULL!");
			//parameter a, b
			CHECK_EQ_OP(_o_args->size(), 1, "scalex parameter must == 1 vs %d",
					_o_args->size());
			CHECK_EQ_OP(s_blobs->size(), 1, "sblobs size must == 1 vs %d",
					s_blobs->size());
		}

		void op(blobs *s_blobs_,blobs *o_blobs_) override {

			o_blobs_ = s_blobs_;

			blob *s_blob_ = (blob*)s_blobs_->at(0);
			cacu_scalex(s_blob_->s_data(), _o_args->at(0), s_blob_->count());
		}

		void grad(blobs *s_blobs_,blobs *o_blobs_) override {

			o_blobs_ = s_blobs_;

			blob *s_blob_ = (blob*)s_blobs_->at(0);
			cacu_scalex(s_blob_->s_diff(), _o_args->at(0), s_blob_->count());
		}

	};

}


#endif
