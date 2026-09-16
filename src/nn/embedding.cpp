#include "embedding.h"

#include <stdexcept>
#include <vector>
#include <cstdlib>

Tensor	Embedding::create_weights(size_t vocab_size, size_t embedding_dim)
{
	std::vector<size_t> shape;

	shape.push_back(vocab_size);
	shape.push_back(embedding_dim);
	return (Tensor(shape));
}

Embedding::Embedding(size_t vocab_size, size_t embedding_dim)
	: _weights(create_weights(vocab_size, embedding_dim), true)
{
	size_t	i;

	i = 0;
	while (i < _weights.value().size())
	{
		_weights.value().data()[i] = 0.01f;
		_weights.gradient().data()[i] = 0.0f;
		i++;
	}
}

Tensor	Embedding::forward(const Tensor& tokens)
{
	std::vector<size_t> shape;
	size_t	i;
	size_t	j;
	size_t	token_id;
	size_t	embedding_dim;

	embedding_dim = _weights.value().shape()[1];

	if (tokens.shape().size() != 1)
		throw (std::invalid_argument("Embedding expects 1D token tensor"));

	shape.push_back(tokens.size());
	shape.push_back(embedding_dim);

	{
		Tensor	result(shape);

		i = 0;
		while (i < tokens.size())
		{
			token_id = static_cast<size_t>(tokens.data()[i]);

			if (token_id >= _weights.value().shape()[0])
				throw (std::out_of_range(
					"Token ID out of vocabulary"));

			j = 0;
			while (j < embedding_dim)
			{
				result.data()[i * embedding_dim + j]
					= _weights.value().data()[token_id * embedding_dim + j];
				j++;
			}
			i++;
		}
		return (result);
	}
}

Variable&	Embedding::weights()
{
	return (_weights);
}
