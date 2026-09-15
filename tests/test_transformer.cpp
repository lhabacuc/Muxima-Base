#include "../src/transformer/transformer.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;

	shape.push_back(4);

	{
		Tensor	tokens(shape);
		Transformer	model(
			32,
			16,
			8,
			2,
			2,
			32);

		tokens.data()[0] = 1.0f;
		tokens.data()[1] = 5.0f;
		tokens.data()[2] = 8.0f;
		tokens.data()[3] = 2.0f;

		Tensor	result = model.forward(tokens);

		if (result.shape()[0] != 4)
			return (1);

		if (result.shape()[1] != 8)
			return (1);
	}

	std::cout << "test_transformer: OK"
		<< std::endl;
	return (0);
}