#include "linear.h"

#include "../core/math.h"

#include <stdexcept>

Tensor	Linear::create_weights(
	size_t input_size,
	size_t output_size)
{
	std::vector<size_t> shape;

	shape.push_back(input_size);
	shape.push_back(output_size);

	Tensor weights(shape);

	size_t i = 0;
	while (i < weights.size())
	{
		weights.data()[i] = 0.01f;
		i++;
	}
	return (weights);
}

Tensor	Linear::create_bias(size_t output_size)
{
	std::vector<size_t> shape;

	shape.push_back(1);
	shape.push_back(output_size);

	Tensor bias(shape);

	size_t i = 0;
	while (i < bias.size())
	{
		bias.data()[i] = 0.0f;
		i++;
	}
	return (bias);
}

Linear::Linear(
	size_t input_size,
	size_t output_size)
	: _weights(create_weights(input_size, output_size), true),
		_bias(create_bias(output_size), true)
{
}

Tensor	Linear::forward(const Tensor& input)
{
	if (input.shape().size() != 2)
		throw std::invalid_argument(
			"Linear input must be 2D");

	return (add(matmul(input, _weights.value()),
		_bias.value()));
}

Variable*	Linear::forward(Variable& input, AutogradGraph& graph)
{
	Variable* hidden = graph.matmul(input, _weights);
	return (graph.add(*hidden, _bias));
}

Variable&	Linear::weights()
{
	return (_weights);
}

Variable&	Linear::bias()
{
	return (_bias);
}
