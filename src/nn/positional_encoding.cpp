#include "positional_encoding.h"

#include <cmath>
#include <vector>
#include <stdexcept>

static double	power(double base, double exponent)
{
	return (std::pow(base, exponent));
}

Tensor	PositionalEncoding::create_encoding(
	size_t max_length,
	size_t embedding_dim)
{
	std::vector<size_t> shape;

	shape.push_back(max_length);
	shape.push_back(embedding_dim);

	{
		Tensor	result(shape);
		size_t	position;
		size_t	dimension;
		double	angle;

		position = 0;
		while (position < max_length)
		{
			dimension = 0;
			while (dimension < embedding_dim)
			{
				angle = static_cast<double>(position)
					/ power(10000.0,
						static_cast<double>(dimension)
						/ static_cast<double>(embedding_dim));

				if (dimension % 2 == 0)
					result.data()[position * embedding_dim + dimension]
						= static_cast<float>(std::sin(angle));
				else
					result.data()[position * embedding_dim + dimension]
						= static_cast<float>(std::cos(angle));

				dimension++;
			}
			position++;
		}
		return (result);
	}
}

PositionalEncoding::PositionalEncoding(
	size_t max_length,
	size_t embedding_dim)
	: _encoding(create_encoding(max_length, embedding_dim))
{
}

Tensor	PositionalEncoding::forward(const Tensor& input)
{
	size_t	sequence_length;
	size_t	embedding_dim;
	size_t	i;
	size_t	j;

	if (input.shape().size() != 2)
		throw (std::invalid_argument(
			"PositionalEncoding expects 2D tensor"));

	sequence_length = input.shape()[0];
	embedding_dim = input.shape()[1];

	if (sequence_length > _encoding.shape()[0])
		throw (std::out_of_range(
			"Sequence exceeds maximum length"));

	if (embedding_dim != _encoding.shape()[1])
		throw (std::invalid_argument(
			"Embedding dimensions do not match"));

	{
		Tensor	result(input.shape());

		i = 0;
		while (i < sequence_length)
		{
			j = 0;
			while (j < embedding_dim)
			{
				result.data()[i * embedding_dim + j]
					= input.data()[i * embedding_dim + j]
					+ _encoding.data()[i * embedding_dim + j];
				j++;
			}
			i++;
		}
		return (result);
	}
}

Tensor&	PositionalEncoding::encoding()
{
	return (_encoding);
}
