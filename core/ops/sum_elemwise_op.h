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

	class sum_elemwise_op : public operator_base
	{

	public:

		sum_elemwise_op(blobs *&data, args *&args) : operator_base(data, args){
			check();

			blob_base *_blob = data->at(0);
			o_blob = cacu_allocator::create_blob(_blob->num(), _blob->channel(), _blob->width(), _blob->height(), _phrase);
		};

		~sum_elemwise_op(){
			delete (blob *)o_blob;
		};

		virtual const void check() override{

			CHECK_GT_OP(((blob*)s_blobs)->num(),0);
			return;
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;

			for (unsigned int j = 0; j < (s_blobs)->size(); ++j){
				blob *s_blob_ = (blob*)(*s_blobs)[j];
				//CHECK_EQ_OP(s_blob_->count(), o_blob->count());
				cacu_saxpy(s_blob_->s_data(), (float_t)1, o_blob_->s_data(), o_blob_->count());
			}
			//echo();
			return;
		}

		virtual const void grad() override{
			blob *o_blob_ = (blob*)o_blob;

			for (unsigned int j = 0; j < (s_blobs)->size(); ++j){
				blob *s_blob_ = (blob*)(*s_blobs)[j];
				//CHECK_EQ_OP(s_blob_->count(), o_blob->count());
				cacu_copy(o_blob_->s_diff(),o_blob_->count(),s_blob_->s_diff());
			}
			//echo();
			return;

		}

		virtual const void load(std::ifstream& is) override{

		}

		virtual const void save(std::ostream& os) override{

		}

		virtual const void echo() override
		{
			//LOG_INFO("%f", ((blob*)o_blob)->s_data()[0]);
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();
		}

	private:

		
	};
};
