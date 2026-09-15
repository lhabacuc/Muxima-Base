#include "../src/nn/linear.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;
	Linear	layer(3, 2);
	Tensor	input;

	shape.push_back(1);
	shape.push_back(3);

	input = Tensor(shape);

	input.data()[0] = 1.0f;
	input.data()[1] = 2.0f;
	input.data()[2] = 3.0f;

	layer.weights().data()[0] = 1.0f;
	layer.weights().data()[1] = 2.0f;
	layer.weights().data()[2] = 3.0f;
	layer.weights().data()[3] = 4.0f;
	layer.weights().data()[4] = 5.0f;
	layer.weights().data()[5] = 6.0f;

	layer.bias().data()[0] = 1.0f;
	layer.bias().data()[1] = 2.0f;

	{
		Tensor	output = layer.forward(input);

		if (output.data()[0] != 23.0f)
			return (1);
		if (output.data()[1] != 30.0f)
			return (1);
	}

	std::cout << "test_linear: OK" << std::endl;
	return (0);
}
