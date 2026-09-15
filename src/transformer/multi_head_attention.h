#ifndef MULTI_HEAD_ATTENTION_H
# define MULTI_HEAD_ATTENTION_H

# include "../nn/linear.h"

# include "../core/math.h"

# include <cstddef>

class MultiHeadAttention
{
	private:
		Linear		_query;
		Linear		_key;
		Linear		_value;
		Linear		_output;
		size_t		_num_heads;
		size_t		_head_dim;

		Tensor	create_head(const Tensor& tensor, size_t head) const;
		Tensor	create_causal_mask(size_t sequence_length) const;
		Tensor	apply_mask(const Tensor& scores,
			const Tensor& mask) const;

	public:
		MultiHeadAttention(size_t embedding_dim, size_t num_heads);

		Tensor	forward(const Tensor& input);

		Linear&	query();
		Linear&	key();
		Linear&	value();
		Linear&	output();

		size_t	num_heads() const;
		size_t	head_dim() const;
};

#endif
