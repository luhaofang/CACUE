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

	class feature_combine_op : public operator_base
	{

	public:

		feature_combine_op(blob_base *&data, args *&args_) : operator_base(data, args_){
			check();
			_units_count = args_->at(0);
#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(data->num()/_units_count, s_blob->channel()*_units_count, s_blob->height(), s_blob->width(), _phrase);
#else
			o_blob = create_oblob(data->num()/_units_count, s_blob->channel()*_units_count, s_blob->height(), s_blob->width(), _phrase);
#endif
			o_blob->_CHECK_SIZE_EQ(s_blob);
			echo();
		};

		~feature_combine_op(){

		};

		virtual const void check() override{
			int mod = s_blob->num() % _args->at(0);
			CHECK_EQ_OP(mod, 0, "s_blob num must be integral multiple of units count vs %d!" , mod);
		}

		virtual const void op() override {
#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;

			int output_num = s_blob->num() / _units_count;
			for(int i = 0 ; i < output_num ;++i)
			{
				for(int j = 0 ; j < _units_count ; ++j)
				{
					cacu_copy_cpu(s_blob_->p_data(i*_units_count+j), s_blob_->length(), o_blob_->p_data(i)+j*s_blob_->length());
				}
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			int output_num = s_blob->num() / _units_count;
			for(int i = 0 ; i < output_num ;++i)
			{
				for(int j = 0 ; j < _units_count ; ++j)
				{
					cacu_copy(s_blob_->p_data(i*_units_count+j), s_blob_->length(), o_blob_->p_data(i)+j*s_blob_->length());
				}
			}
#endif
		}

		virtual const void grad() override{

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blob;

			int output_num = s_blob->num() / _units_count;
			for(int i = 0 ; i < output_num ;++i)
			{
				for(int j = 0 ; j < _units_count ; ++j)
				{
					cacu_copy(o_blob_->p_diff_d(i)+j*s_blob_->length(), s_blob_->length(), s_blob_->p_diff_d(i*_units_count+j));
					s_blob_->_sync(i*_units_count+j);
				}
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			int output_num = s_blob->num() / _units_count;
			for(int i = 0 ; i < output_num ;++i)
			{
				for(int j = 0 ; j < _units_count ; ++j)
				{
					cacu_copy(o_blob_->p_diff(i)+j*s_blob_->length(), s_blob_->length(), s_blob_->p_diff(i*_units_count+j));
				}
			}
#endif
		}

		virtual const void load(std::ifstream& is) override{
			return;
		}

		virtual const void save(std::ostream& os) override{
			return;
		}

		virtual const void echo() override
		{
			LOG_INFO("create feature combine op:");
			LOG_INFO("channel: %d, output_channel: %d",s_blob->channel(),o_blob->channel());
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{

			o_blob->_RESET_DATA();

		}

	private:

		//combine unit counts
		int _units_count;
	};
};
