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

	class network
	{
	public:

		network(blobs *&datas_){
			_input_blobs = datas_;
		};

		~network(){
			delete _input_blobs;
		};

		/*
		 * Class network unlike layer & layer_block,
		 * it maintains the abstract layer's operator entity.
		 *
		 */
		network& operator <<(layer_block* const &layer_block_) {
			for (int i = 0; i < layer_block_->length(); ++i){
				_layers.push_back(layer_block_->layer_bases(i));
				for(int j = 0; j<layer_block_->layer_bases(i)->op_count();++j)
					_ops.push_back(layer_block_->layer_bases(i)->op(j));
			}
			return *this;
		}

		network& operator <<(layer_block &layer_block_) {
			for (int i = 0; i < layer_block_.length(); ++i){
				_layers.push_back(layer_block_.layer_bases(i));
				for(int j = 0; j<layer_block_.layer_bases(i)->op_count();++j)
					_ops.push_back(layer_block_.layer_bases(i)->op(j));
			}
			return *this;
		}

		network& operator <<(layer_base* const &layer_) {

			_layers.push_back(layer_);
			for(int j = 0; j<layer_->op_count();++j)
				_ops.push_back(layer_->op(j));
			return *this;
		}

		network& operator <<(layer_base &layer_) {

			_layers.push_back(&layer_);
			for(int j = 0; j<layer_.op_count();++j)
				_ops.push_back(layer_.op(j));
			return *this;
		}

		inline layer *&layers(int i){ return (layer*&)_layers[i]; }

		inline layer_base *&layer_bases(int i){ return _layers[i]; }

		inline int layer_count(){ return _layers.size(); }

		inline void set_inputdata(blob_base *&blob_){  layers(0)->get_head_op()->set_blob(blob_);}

		inline void predict(){
			for(unsigned int i = 0 ; i < _layers.size();++i){
				clock_t start = clock();
				_layers[i]->operate();
				clock_t end = clock();
				//LOG_INFO("%d time cost: %d", i ,end - start);
			}
		}

		inline void back_propagate(){
			for(int i = op_count() - 1 ; i >= 0; --i)
			{
				get_op(i)->grad();
			}
		}

		inline void set_weights_type(param_init_type type_,float_t value = 0){
			for(int i = op_count() - 1 ; i >= 0; --i)
			{
				for(int j = 0 ; j < get_op(i)->weights_size(); ++j)
				{
					get_op(i)->get_weight(j)->set_init_type(type_,value);
				}
			}
		}

		inline operator_base *&get_op(int i)
		{
			return _ops[i];
		}

		inline int op_count(){ return _ops.size(); }

		inline blobs *& input_blobs(){ return _input_blobs; }

		inline blob *& output_blob(){return _ops[_ops.size()-1]->out_data<blob>();}

		inline void output_blobs(){
			for(int i = 0; i < op_count(); ++i)	{
				blob_base * bp = get_op(i)->out_data<blob_base>();
				if(bp != NULL&&bp->is_output())
					LOG_DEBUG("%d",i);
				else if(bp == NULL)
				{
					for(int j = 0 ; j < get_op(i)->out_datas()->size(); ++j)
					{
						if(get_op(i)->out_datas()->at(j)->is_output())
							LOG_DEBUG("%d:%d",i,j);
					}
				}
			}
		}

		inline phrase_type phrase(){return _input_blobs->at(0)->phrase();}

		void load_weights(chars_t modelpath){
			std::ifstream is(modelpath, ios::binary);
			is.precision(std::numeric_limits<float_t>::digits10);
			for(int i = 0; i < op_count(); ++i)	{
				get_op(i)->load(is);
			}
			is.close();
			LOG_INFO("Initialize model by : %s",modelpath.c_str());
		}

		void save_weights(chars_t modelpath){
			std::ofstream os(modelpath, ios::binary);
			os.precision(std::numeric_limits<float_t>::digits10);
			for(int i = 0; i < op_count(); ++i)	{
				get_op(i)->save(os);
			}
			os.close();
			LOG_INFO("optimization is done!");
			LOG_INFO("The model is saved at : %s",modelpath.c_str());
		}

		void check(){
			for(unsigned int i =0 ; i < _ops.size();++i){
				clock_t start = clock();
				_ops[i]->echo();
				clock_t end = clock();
				//LOG_INFO("%d time cost: %d", i ,end - start);
			}
		}

	private:

		vector<layer_base*> _layers;

		vector<operator_base*> _ops;

		blobs* _input_blobs;

	};

};
