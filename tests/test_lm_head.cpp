#include "../src/nn/lm_head.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;

	shape.push_back(3);
	shape.push_back(8);

	{
		Tensor	input(shape);
		LMHead	head(8, 32);

		input.data()[0] = 1.0f;
		input.data()[1] = 2.0f;

		Tensor	result = head.forward(input);

		if (result.shape()[0] != 3)
			return (1);

		if (result.shape()[1] != 32)
			return (1);
	}

	std::cout << "test_lm_head: OK" << std::endl;
	return (0);
}