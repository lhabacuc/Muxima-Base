#ifndef LM_HEAD_H
# define LM_HEAD_H

# include "../core/tensor.h"
# include "linear.h"

class LMHead
{
	private:
		size_t	_embedding_dim;
		size_t	_vocab_size;
		Linear	_projection;

	public:
		LMHead(size_t embedding_dim, size_t vocab_size);

		Tensor	forward(const Tensor& input);

		Linear&	projection();
};

#endif