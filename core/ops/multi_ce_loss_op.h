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

	class multi_ce_loss_op : public operator_base
	{

	public:

		multi_ce_loss_op(blobs *&data, args *&args_) : operator_base(data, args_, CACU_SOFTMAX_LOSS){
			check();

			initial(data,args_);
			init_weights(data,args_);

			_loss = (float_t*)malloc(sizeof(float_t));


			echo();
		};

		~multi_ce_loss_op(){
			free(_loss);
		};

		virtual const void initial(blob_base *&data, args *&args_) override{

#if __USEMBEDDING__ == ON
			o_blob = create_em_oblob(data->num(),args_->output_channel(),1,1,train);
#else
			o_blob = create_oblob(data->num(),args_->output_channel(),1,1,train);
#endif
			_trans_labels = cacu_allocator::create_bin_blob(data->num(), args_->output_channel(), 1, 1, test);
		}

		virtual const void init_weights(blob_base *&data, args *&args_) override{
			return;
		}

		virtual const void check() override{
			CHECK_GT_OP(_args->output_channel(),0,"loss num must > 0 vs %d !",_args->output_channel());
		}

		virtual const void op() override {

			_loss[0] = 0.0;

			bin_blob *labels_ = (bin_blob*)s_blobs->at(1);
			cacu_multi_label_trans(s_blob->num(), _args->output_channel(), labels_->s_data(), _trans_labels->s_data());

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blobs->at(0);

			for(int n = 0; n < s_blob->num(); ++n){
				for(int i = 0; i < _args->output_channel(); ++i){
					//cacu_cross_entropy_cpu(o_blob_->p_data(i),o_blob_->num(),o_blob_->length(),labels_->s_data(),o_blob_->p_diff(i));
					
				}
			}

#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blobs->at(0);

			cacu_cross_entropy(o_blob_->s_data(),o_blob_->num(),o_blob_->length(),labels_->s_data(),o_blob_->s_diff());
#endif

#if __USEMBEDDING__ == ON
			cacu_copy_cpu(o_blob_->s_diff(), 1 ,_loss);
#else
	#if __PARALLELTYPE__ == __CUDA__
			cuda_copy2host(_loss, o_blob_->s_diff(), 1);
	#else
			cacu_copy(o_blob_->s_diff(), 1 ,_loss);
	#endif
#endif
			_loss[0] *= normalizer();
		}

		virtual const void grad() override{

#if __USEMBEDDING__ == ON
			em_blob *o_blob_ = (em_blob*)o_blob;
			em_blob *s_blob_ = (em_blob*)s_blobs->at(0);
			em_bin_blob *labels_ = (em_bin_blob*)s_blobs->at(1);

			//CE LOSS BACK PROPGATION
			for (int i = 0 ; i < s_blob_->num() ; ++i)
			{
				//cacu_isaxb(o_blob_->p_data_d(i),s_blob_->length(),(float_t)1,labels_->p_data_d(i),(float_t)-1, s_blob_->p_diff_d(i));
				cacu_scalex(s_blob_->p_diff_d(i),s_blob_->length(),normalizer());
				s_blob_->_sync(i);
			}
#else
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blobs->at(0);
			bin_blob *labels_ = (bin_blob*)s_blobs->at(1);

			//CE LOSS BACK PROPGATION
			for (int i = 0 ; i < s_blob_->num() ; ++i)
			{
				//cacu_isaxb(o_blob_->p_data(i),s_blob_->length(),(float_t)1,labels_->p_data(i),(float_t)-1, s_blob_->p_diff(i));
				cacu_scalex(s_blob_->p_diff(i),s_blob_->length(),normalizer());
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
			LOG_INFO("loss : %f", _loss[0]);
		}

		inline virtual const void LOOP_INIT_DATA_() override
		{
			o_blob->_RESET_DATA();
		}

		inline virtual const void set_phrase(phrase_type phrase_) override {
			_phrase = phrase_;
		}

		float_t normalizer()
		{
			blob_base* blob_= s_blobs->at(0);
			return ((float_t)(1)/blob_->num());
		}

		inline float_t loss(){return _loss[0];}

	private:

		float_t *_loss;

		bin_blob * _trans_labels;

	};
};
