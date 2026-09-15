#include "linear.h"

#include "../core/math.h"

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
