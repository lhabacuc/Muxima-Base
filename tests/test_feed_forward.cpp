#include "../src/nn/feed_forward.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;

	shape.push_back(3);
	shape.push_back(4);

	{
		Tensor	input(shape);
		Tensor	result;
		FeedForward	ffn(4, 16);

		input.data()[0] = 1.0f;
		input.data()[1] = 2.0f;
		input.data()[2] = 3.0f;
		input.data()[3] = 4.0f;

		result = ffn.forward(input);

		if (result.shape()[0] != 3)
			return (1);

		if (result.shape()[1] != 4)
			return (1);
	}

	std::cout << "test_feed_forward: OK"
		<< std::endl;
	return (0);
}