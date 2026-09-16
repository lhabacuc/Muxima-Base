#include "layer_norm.h"

#include <cmath>
#include <stdexcept>
#include <vector>

Tensor	LayerNorm::create_parameter(size_t embedding_dim)
{
	std::vector<size_t> shape;

	shape.push_back(1);
	shape.push_back(embedding_dim);
	return (Tensor(shape));
}

LayerNorm::LayerNorm(size_t embedding_dim, float eps)
	: _gamma(create_parameter(embedding_dim), true),
	_beta(create_parameter(embedding_dim), true),
	_eps(eps)
{
	size_t	i;

	i = 0;
	while (i < embedding_dim)
	{
		_gamma.value().data()[i] = 1.0f;
		_gamma.gradient().data()[i] = 0.0f;
		_beta.value().data()[i] = 0.0f;
		_beta.gradient().data()[i] = 0.0f;
		i++;
	}
}

Tensor	LayerNorm::forward(const Tensor& input)
{
	Tensor	result(input.shape());
	size_t	sequence_length;
	size_t	embedding_dim;
	size_t	i;
	size_t	j;
	float	mean;
	float	variance;
	float	diff;

	if (input.shape().size() != 2)
		throw (std::invalid_argument(
			"LayerNorm expects 2D tensor"));

	sequence_length = input.shape()[0];
	embedding_dim = input.shape()[1];

	if (embedding_dim != _gamma.value().shape()[1])
		throw (std::invalid_argument(
			"Embedding dimensions do not match"));

	i = 0;
	while (i < sequence_length)
	{
		mean = 0.0f;
		j = 0;
		while (j < embedding_dim)
		{
			mean += input.data()[i * embedding_dim + j];
			j++;
		}
		mean /= static_cast<float>(embedding_dim);

		variance = 0.0f;
		j = 0;
		while (j < embedding_dim)
		{
			diff = input.data()[i * embedding_dim + j] - mean;
			variance += diff * diff;
			j++;
		}
		variance /= static_cast<float>(embedding_dim);

		j = 0;
		while (j < embedding_dim)
		{
			diff = input.data()[i * embedding_dim + j] - mean;
			result.data()[i * embedding_dim + j]
				= diff / std::sqrt(variance + _eps);
			result.data()[i * embedding_dim + j]
				= result.data()[i * embedding_dim + j]
				* _gamma.value().data()[j] + _beta.value().data()[j];
			j++;
		}
		i++;
	}
	return (result);
}

Variable&	LayerNorm::gamma()
{
	return (_gamma);
}

Variable&	LayerNorm::beta()
{
	return (_beta);
}
