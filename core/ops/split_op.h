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

	class split_op : public operator_base
	{

	public:

		split_op(blob_base *&data, args *&args_) : operator_base(data, args_, CACU_SPLIT){
			check();

			int split_count = args_->at(0);
			o_blobs = create_oblobs();
			for(int i = 0 ; i < split_count ; ++i)
			{
	#if __USEMBEDDING__ == ON
				o_blobs->push_back(cacu_allocator::create_em_blob(data->num(),data->channel(),data->height(),data->width(), 0,_phrase));
	#else
				o_blobs->push_back(cacu_allocator::create_blob(data->num(),data->channel(),data->height(),data->width(), 0,_phrase));
	#endif
			}

			echo();
		};

		~split_op(){

		};

		virtual const void check() override{
			//split count > 0
			CHECK_GT_OP(_args->at(0), 0,"output_channel must > 0 vs %d",_args->at(0));
		}

		virtual const void op() override {

#if __USEMBEDDING__ == ON
			em_blob *s_blob_ = (em_blob*)s_blob;

			for (int j = 0; j < (o_blobs)->size(); ++j){
				em_blob *o_blob_ = (em_blob *)o_blobs->at(j);
				cacu_copy_cpu(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
			}
#else
			blob *s_blob_ = (blob*)s_blob;

			for (unsigned int j = 0; j < (o_blobs)->size(); ++j){
				blob *o_blob_ = (blob *)o_blobs->at(j);
				cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
			}
#endif
		}

		virtual const void grad() override{

#if __USEMBEDDING__ == ON
			em_blob *s_blob_ = (em_blob*)s_blob;

			for(int i = 0 ; i < s_blob_->num();++i){
				for (int j = 0; j < (o_blobs)->size(); ++j){
					em_blob *o_blob_ = (em_blob *)o_blobs->at(j);
					cacu_saxpy(o_blob_->p_diff_d(i),(float_t)1,s_blob_->p_diff_d(i),o_blob_->length());
				}
				s_blob_->_sync(i);
			}
#else
			blob *s_blob_ = (blob*)s_blob;

			for (unsigned int j = 0; j < (o_blobs)->size(); ++j){
				blob *o_blob_ = (blob *)o_blobs->at(j);
				cacu_saxpy(o_blob_->s_diff(),(float_t)1,s_blob_->s_diff(),o_blob_->count());
			}
#endif
		}

		virtual const void load(std::ifstream& is) override{
			return;
		}

		virtual const void save(std::ostream& os) override{
			return;
		}

		virtual const void echo() override{
			LOG_INFO("create split op:");
			LOG_INFO("channel: %d, input_dim: %d, output_channel: %d, output_dim: %d",s_blob->channel(),s_blob->height(),o_blobs->at(0)->channel(),o_blobs->at(0)->height());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blobs->_RESET_DATA();
		}

		inline virtual const void set_phrase(phrase_type phrase_) override {
			_phrase = phrase_;
		}

	private:


	};
};
