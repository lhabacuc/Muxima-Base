#ifndef POSITIONAL_ENCODING_H
# define POSITIONAL_ENCODING_H

# include "../core/tensor.h"

class PositionalEncoding
{
	private:
		Tensor	_encoding;

		static Tensor	create_encoding(size_t max_length,
						size_t embedding_dim);

	public:
		PositionalEncoding(size_t max_length, size_t embedding_dim);

		Tensor	forward(const Tensor& input);

		Tensor&	encoding();
};

#endif
