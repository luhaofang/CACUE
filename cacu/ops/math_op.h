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

#ifndef MATH_OP_H_
#define MATH_OP_H_

namespace cacu {

	class math_op : public operator_base {

	public:

		math_op(blobs *&data, op_name op_name_) :
			operator_base(data, op_name_) {
			_INIT_OP();
		}

		math_op(blobs *&data, op_args *&_args, op_name op_name_) :
			operator_base(data, _args, op_name_) {
			_INIT_OP();
		}


		~math_op() {

		}

		void initial() override {
			if (o_blobs == NULL){
				o_blobs = create_oblobs();
				o_blobs->push_back(create_oblob(s_blobs->at(0)->num(),
						s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(), _phase));
			}
			else
				o_blobs->at(0)->resize(s_blobs->at(0)->num(), s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height());
		}

		void init_weights() override {
			return;
		}

		void check() override {
			return;
		}

		void load(std::ifstream& is) final override {
			return;
		}

		void save(std::ostream& os) final override {
			return;
		}

		void echo() override {

			if (_FUNC == MASK_LT) {
				blob *o_blob_ = (blob*)o_blobs->at(0);
				LOG_INFO("Dimension loss: %f", o_blob_->s_diff()[0]);
			}
			else if (_FUNC == MASK_GT){
				blob *o_blob_ = (blob*)o_blobs->at(0);
				LOG_INFO("Spasity loss: %f", o_blob_->s_diff()[0]);
			}
			else {
				LOG_INFO("create math op:");
				LOG_INFO(
					"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
					s_blobs->at(0)->channel(), s_blobs->at(0)->width(), s_blobs->at(0)->height(),
					o_blobs->at(0)->channel(), o_blobs->at(0)->width(), o_blobs->at(0)->height());
			}
		}

	private:

		math_op_name _FUNC = MASK_GT;

	};

}


#endif
