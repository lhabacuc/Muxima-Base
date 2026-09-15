#ifndef SELF_ATTENTION_H
# define SELF_ATTENTION_H

# include "../core/tensor.h"
# include "../nn/linear.h"

class SelfAttention
{
	private:
		size_t	_embedding_dim;
		size_t	_head_dim;

		Linear	_query;
		Linear	_key;
		Linear	_value;
		Linear	_output;

	public:
		SelfAttention(size_t embedding_dim);

		Tensor	forward(const Tensor& input);

		Linear&	query();
		Linear&	key();
		Linear&	value();
		Linear&	output();
};

#endif
