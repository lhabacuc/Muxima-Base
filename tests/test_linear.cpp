#include "../src/nn/linear.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> input_shape;
	Linear	layer(3, 2);

	input_shape.push_back(1);
	input_shape.push_back(3);

	Tensor	input(input_shape);

	input.data()[0] = 1.0f;
	input.data()[1] = 2.0f;
	input.data()[2] = 3.0f;

	layer.weights().value().data()[0] = 1.0f;
	layer.weights().value().data()[1] = 2.0f;
	layer.weights().value().data()[2] = 3.0f;
	layer.weights().value().data()[3] = 4.0f;
	layer.weights().value().data()[4] = 5.0f;
	layer.weights().value().data()[5] = 6.0f;

	layer.bias().value().data()[0] = 1.0f;
	layer.bias().value().data()[1] = 2.0f;

	{
		Tensor	output = layer.forward(input);

		if (output.shape()[0] != 1)
			return (1);
		if (output.shape()[1] != 2)
			return (1);
		if (output.data()[0] != 23.0f)
			return (1);
		if (output.data()[1] != 30.0f)
			return (1);
	}

	std::cout << "test_linear: OK" << std::endl;
	return (0);
}
