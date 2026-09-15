#ifndef EMBEDDING_H
# define EMBEDDING_H

# include "../core/tensor.h"

class Embedding
{
	private:
		Tensor	_weights;

		static Tensor	create_weights(size_t vocab_size,
						size_t embedding_dim);

	public:
		Embedding(size_t vocab_size, size_t embedding_dim);

		Tensor	forward(const Tensor& tokens);

		Tensor&	weights();
};

#endif
