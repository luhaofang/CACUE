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

		network(){
		
#if  __PARALLELTYPE__ == __GPU__
			cublasCreate_v2(&handle);
#endif

		};

		~network(){

#if  __PARALLELTYPE__ == __GPU__
			cublasDestroy_v2(handle);
#endif

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

		inline void predict(blob_base *blob_){
			for(unsigned int i =0 ; i < _layers.size();++i){
				clock_t start = clock();
				_layers[i]->operate();
				clock_t end = clock();
				LOG_INFO("%d time cost:%d", i ,end - start);
			}
		}

		inline operator_base *&get_op(int i)
		{
			return _ops[i];
		}

		inline int op_count(){ return _ops.size(); }



	private:

		vector<layer_base*> _layers;

		vector<operator_base*> _ops;

		//blobs* _input_data;
	};

};
