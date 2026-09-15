#include "../src/transformer/transformer_block.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;

	shape.push_back(3);
	shape.push_back(4);

	{
		Tensor	input(shape);
		TransformerBlock	block(4, 2, 16);

		input.data()[0] = 1.0f;
		input.data()[1] = 2.0f;
		input.data()[2] = 3.0f;
		input.data()[3] = 4.0f;

		Tensor	result = block.forward(input);

		if (result.shape()[0] != 3)
			return (1);

		if (result.shape()[1] != 4)
			return (1);
	}

	std::cout << "test_transformer_block: OK"
		<< std::endl;
	return (0);
}