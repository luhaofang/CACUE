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

#ifndef MATH_OPERATOR_BASE_H_
#define MATH_OPERATOR_BASE_H_

namespace cacu {

	class math_op : public operator_base {

	public:

		math_op(blob_base *&data) :
			operator_base(data, CACU_MATH) {
			check();
			initial();
			init_weights();
			echo();
		}

		~math_op() {

		}

		virtual const void initial() override {
			if (o_blob == NULL)
				o_blob = create_oblob(s_blob->num(), s_blob->channel(), s_blob->width(), s_blob->height(), _phase);
			else
				o_blob->resize(s_blob->num(), s_blob->channel(), s_blob->width(), s_blob->height());
		}

		virtual const void init_weights() override {
			return;
		}

		virtual const void check() override {
			return;
		}

		virtual const void op() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			switch (_FUNC)
			{
			case ADD:
				s_blob->_CHECK_SIZE_EQ(_data);
				cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				cacu_saxpy(o_blob_->s_data(), (float_t)1, _data->s_data(), _data->count());
				break;
			case SUB:
				s_blob->_CHECK_SIZE_EQ(_data);
				cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				cacu_saxpy(_data->s_data(), (float_t)-1, o_blob_->s_data(), _data->count());
				break;
			case MUL:
				//s_blob->_CHECK_SIZE_EQ(_data);
				cacu_copy(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				for (int i = 0; i < s_blob->num(); ++i)
				{
					for (int c = 0; c < s_blob->channel(); ++c)
						cacu_ssx(_data->s_data(), s_blob_->channel_length(),
							o_blob_->p_data(i) + c * s_blob_->channel_length());
				}
				//cacu_print(o_blob_->s_data(), 10);
				//cacu_print(_data->s_data(), 1000);
				break;
			case DIV:
				break;
			case ABS:
				cacu_abs(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				break;
			case MASK_LT:
				mask_vector_lt(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				cacu_ssx(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				cacu_sumbysize(BYWIDTH, o_blob_->s_data(), o_blob_->count(), -1.0 / o_blob_->num(), o_blob_->s_diff(), 0, o_blob_->count());
				break;
			case MASK_GT:
				mask_vector(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				cacu_ssx(s_blob_->s_data(), s_blob_->count(), o_blob_->s_data());
				cacu_sumbysize(BYWIDTH, o_blob_->s_data(), o_blob_->count(), 1.0 / o_blob_->num(), o_blob_->s_diff(), 0, o_blob_->count());
				break;
			default:
				break;
			}

		}

		virtual const void grad() override {
			blob *o_blob_ = (blob*)o_blob;
			blob *s_blob_ = (blob*)s_blob;

			switch (_FUNC)
			{
			case ADD:
				s_blob->_CHECK_SIZE_EQ(_data);
				cacu_saxpy(o_blob_->s_diff(), (float_t)1, s_blob_->s_diff(), o_blob_->count());
				cacu_saxpy(o_blob_->s_diff(), (float_t)1, _data->s_diff(), o_blob_->count());
				break;
			case SUB:
				s_blob->_CHECK_SIZE_EQ(_data);
				cacu_saxpy(o_blob_->s_diff(), (float_t)1, s_blob_->s_diff(), o_blob_->count());
				cacu_saxpy(o_blob_->s_diff(), (float_t)-1, _data->s_diff(), o_blob_->count());
				break;
			case MUL:
				cacu_copy(s_blob_->s_data(), s_blob_->count(), s_blob_->s_diff());
				for (int i = 0; i < s_blob->num(); ++i)
				{
					for (int c = 0; c < s_blob->channel(); ++c)
						cacu_ssx(o_blob_->p_diff(i) + c * s_blob_->channel_length(), s_blob_->channel_length(),
							s_blob_->p_diff(i) + c * s_blob_->channel_length());
				}
				cacu_sumbysize(BYHEIGHT, s_blob_->s_diff(), s_blob_->count(), 1., _data->s_diff(), 0., _data->count());
				break;
			case DIV:
				break;
			case ABS:
				cacu_abs_grad(s_blob_->s_data(), s_blob_->s_diff(), s_blob_->count(), o_blob_->s_diff());
				break;
			case MASK_LT:
				mask_vector_lt(s_blob_->s_data(), s_blob_->count(), s_blob_->s_diff());
				cacu_scalex(s_blob_->s_diff(), s_blob_->count(), (float_t)1.0 / s_blob_->num());
				break;
				//cacu_print(s_blob_->s_data(), 10);
				//cacu_print(s_blob_->s_diff(), 10);
			case MASK_GT:
				mask_vector(s_blob_->s_data(), s_blob_->count(), s_blob_->s_diff());
				cacu_scalex(s_blob_->s_diff(), s_blob_->count(), (float_t)1.0 / s_blob_->num());
				break;
				//cacu_print(s_blob_->s_data(), 10);
				//cacu_print(s_blob_->s_diff(), 10);
			default:
				break;
			}
		}

		virtual const void load(std::ifstream& is) override {
			return;
		}

		virtual const void save(std::ostream& os) override {
			return;
		}

		virtual const void echo() override {

			if (_FUNC == MASK_LT) {
				blob *o_blob_ = (blob*)o_blob;
				LOG_INFO("Dimension loss: %f", o_blob_->s_diff()[0]);
			}
			else if (_FUNC == MASK_GT){
				blob *o_blob_ = (blob*)o_blob;
				LOG_INFO("Spasity loss: %f", o_blob_->s_diff()[0]);
			}
			else {
				LOG_INFO("create math op:");
				LOG_INFO(
					"channel: %d, input_dim: (%d,%d), output_channel: %d, output_dim: (%d,%d)",
					s_blob->channel(), s_blob->width(), s_blob->height(),
					o_blob->channel(), o_blob->width(), o_blob->height());
			}
		}

		inline virtual const void LOOP_INIT_DATA_() override {
			return;
		}

		inline virtual const void set_phase(phase_type phase_) override {
			_phase = phase_;
		}

		void function(math_op_name function, blob *data_= NULL)
		{
			this->_FUNC = function;
			this->_data = data_;
		}

	private:

		math_op_name _FUNC;

		blob *_data;


	};

}


#endif
