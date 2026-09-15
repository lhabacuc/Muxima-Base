#include "linear.h"

#include "../core/math.h"

static Tensor	create_weights(size_t input_size, size_t output_size)
{
	std::vector<size_t> shape;

	shape.push_back(input_size);
	shape.push_back(output_size);
	return (Tensor(shape));
}

static Tensor	create_bias(size_t output_size)
{
	std::vector<size_t> shape;

	shape.push_back(1);
	shape.push_back(output_size);
	return (Tensor(shape));
}

Linear::Linear(size_t input_size, size_t output_size)
	: _weights(create_weights(input_size, output_size)),
	_bias(create_bias(output_size))
{
	size_t	i;

	i = 0;
	while (i < _weights.size())
	{
		_weights.data()[i] = 0.01f;
		i++;
	}

	i = 0;
	while (i < _bias.size())
	{
		_bias.data()[i] = 0.0f;
		i++;
	}
}

Tensor	Linear::forward(const Tensor& input)
{
	return (matmul(input, _weights));
}

Tensor&	Linear::weights()
{
	return (_weights);
}

Tensor&	Linear::bias()
{
	return (_bias);
}
