#include "../src/nn/layer_norm.h"

#include <cmath>
#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;
	LayerNorm	norm(4, 1e-5f);

	shape.push_back(1);
	shape.push_back(4);

	{
		Tensor	input(shape);
		Tensor	result(shape);
		float	expected;

		input.data()[0] = 1.0f;
		input.data()[1] = 2.0f;
		input.data()[2] = 3.0f;
		input.data()[3] = 4.0f;

		result = norm.forward(input);

		expected = -1.3416f;
		if (std::fabs(result.data()[0] - expected) > 0.001f)
			return (1);

		expected = -0.4472f;
		if (std::fabs(result.data()[1] - expected) > 0.001f)
			return (1);

		expected = 0.4472f;
		if (std::fabs(result.data()[2] - expected) > 0.001f)
			return (1);

		expected = 1.3416f;
		if (std::fabs(result.data()[3] - expected) > 0.001f)
			return (1);
	}

	std::cout << "test_layer_norm: OK" << std::endl;
	return (0);
}