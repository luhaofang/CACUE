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

#include "../utils/cacu_allocator.h"

namespace mycnn{

	class operator_base
	{

	public:

		operator_base(blob_base *&data, args *&args_){

			s_blob = data;
			s_blobs = NULL;
			o_blob = NULL;
			o_blobs = NULL;
			_args = args_;
			_phrase = data->phrase();
			data->_REC();
		};

		operator_base(blob_base *&data){

			s_blob = data;
			s_blobs = NULL;
			o_blob = NULL;
			o_blobs = NULL;
			_args = NULL;
			_phrase = data->phrase();
			data->_REC();
		};

		operator_base(blobs *&data, args *&args_){

			s_blob = NULL;
			s_blobs = data;
			o_blob = NULL;
			o_blobs = NULL;
			_args = args_;
			_phrase = data->at(0)->phrase();
			data->_REC();
		};

		operator_base(blobs *&data){

			s_blob = NULL;
			s_blobs = data;
			o_blob = NULL;
			o_blobs = NULL;
			_args = NULL;
			_phrase = data->at(0)->phrase();
			data->_REC();
		};

		virtual ~operator_base(){

			delete _args;
			if(_IS_ALLOC_OUTPUT){
				if(o_blob != NULL)
					delete o_blob;
				if(o_blobs != NULL)
					delete o_blobs;
			}
			for(unsigned int i = 0 ; i< _weights.size(); ++i)
			{
				delete _weights[i];
				_weights[i] = NULL;
			}
		};

		virtual const void op() = 0;

		virtual const void check() = 0;

		virtual const void grad() = 0;

		virtual const void load(std::ifstream &is) = 0;

		virtual const void save(std::ostream &os) = 0;

		virtual const void echo() = 0;

		virtual const void LOOP_INIT_DATA_() = 0;

		virtual const void set_phrase(phrase_type phrase_) = 0;

		inline blobs *&out_datas(){ return o_blobs; }

		template<typename BTYPE>
		inline BTYPE *&out_data(){ return (BTYPE *&)o_blob; }

		inline blobs *&in_datas(){ return s_blobs; }

		template<typename BTYPE>
		inline BTYPE *&in_data(){return (BTYPE *&)s_blob;}

		inline int weights_size(){ return _weights.size(); }

		inline weight* get_weight(int i){ return _weights[i]; }

		inline void infer()
		{
			//reset the data's values
			LOOP_INIT_DATA_();
			//forward propagation
			op();
		}

		inline void set_blob(blob_base *&blob_){ s_blob = blob_;}

		inline void set_blobs(blobs *&blobs_){ s_blobs = blobs_;}





	protected:

		blobs *s_blobs;

		blob_base *s_blob;

		blob_base *o_blob;

		blobs *o_blobs;

		args *_args;

		phrase_type _phrase;

		vector<weight*> _weights;

		//create weight push_back to weights container
		inline weight* const create_param(chars_t name,int num,int channel,int width,int height,phrase_type phrase)
		{
			_add2op_weights(new weight(name,num,channel,width,height,phrase));
			return _weights.back();
		}

		inline void set_param_init_type(param_init_type type, weight *w_, float_t value = 0.0)
		{
			w_->set_init_type(type,value);
		}

		inline void _add2op_weights(weight *w){_weights.push_back(w);}

		inline blob_base * create_oblob(int num,int channel, int height, int width, phrase_type phrase_){
			_IS_ALLOC_OUTPUT = true;
			return new blob(num, channel,height,width, 0 ,phrase_);
		}

#if __USEMBEDDING__ == ON
		inline blob_base * create_em_oblob(int num,int channel, int height, int width, phrase_type phrase_){
			_IS_ALLOC_OUTPUT = true;
			return new em_blob(num, channel,height,width, 0 ,phrase_);
		}
#endif
		inline blobs * create_oblobs(){
			_IS_ALLOC_OUTPUT = true;
			return new blobs();
		}

	private:

		bool _IS_ALLOC_OUTPUT = false;

	};
}
