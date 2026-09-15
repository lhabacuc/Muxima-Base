#include "../src/nn/positional_encoding.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;
	PositionalEncoding positional(16, 4);

	shape.push_back(3);
	shape.push_back(4);

	Tensor	input(shape);

	input.data()[0] = 1.0f;
	input.data()[1] = 1.0f;
	input.data()[2] = 1.0f;
	input.data()[3] = 1.0f;

	input.data()[4] = 1.0f;
	input.data()[5] = 1.0f;
	input.data()[6] = 1.0f;
	input.data()[7] = 1.0f;

	input.data()[8] = 1.0f;
	input.data()[9] = 1.0f;
	input.data()[10] = 1.0f;
	input.data()[11] = 1.0f;

	{
		Tensor	result = positional.forward(input);

		if (result.shape()[0] != 3)
			return (1);
		if (result.shape()[1] != 4)
			return (1);

		if (result.data()[0] != 1.0f)
			return (1);

		if (result.data()[1] != 2.0f)
			return (1);

		if (result.data()[4] == 1.0f)
			return (1);
	}

	std::cout << "test_positional_encoding: OK" << std::endl;
	return (0);
}