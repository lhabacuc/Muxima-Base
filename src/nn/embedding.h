#ifndef EMBEDDING_H
# define EMBEDDING_H

# include "../core/variable.h"

class Embedding
{
	private:
		Variable	_weights;

		static Tensor	create_weights(size_t vocab_size,
						size_t embedding_dim);

	public:
		Embedding(size_t vocab_size, size_t embedding_dim);

		Tensor	forward(const Tensor& tokens);

		Variable&	weights();
};

#endif
