#ifndef MULTI_HEAD_ATTENTION_H
# define MULTI_HEAD_ATTENTION_H

# include "../core/tensor.h"
# include "../nn/linear.h"

class MultiHeadAttention
{
	private:
		size_t	_embedding_dim;
		size_t	_num_heads;
		size_t	_head_dim;

		Linear	_query;
		Linear	_key;
		Linear	_value;
		Linear	_output;

		Tensor	attention(
			const Tensor& input,
			size_t head);

		Tensor	concatenate_heads(
			const std::vector<Tensor>& heads);

	public:
		MultiHeadAttention(
			size_t embedding_dim,
			size_t num_heads);

		Tensor	forward(const Tensor& input);

		Linear&	query();
		Linear&	key();
		Linear&	value();
		Linear&	output();
};

#endif
