#include "../src/nn/embedding.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> token_shape;
	Embedding	embedding(10, 4);

	token_shape.push_back(3);

	Tensor	tokens(token_shape);

	tokens.data()[0] = 2.0f;
	tokens.data()[1] = 5.0f;
	tokens.data()[2] = 7.0f;

	embedding.weights().data()[8] = 42.0f;

	{
		Tensor	result = embedding.forward(tokens);

		if (result.shape()[0] != 3)
			return (1);
		if (result.shape()[1] != 4)
			return (1);

		if (result.data()[0] != 42.0f)
			return (1);

		if (result.data()[8] != 0.01f)
			return (1);
	}

	std::cout << "test_embedding: OK" << std::endl;
	return (0);
}
