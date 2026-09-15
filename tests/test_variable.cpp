#include "../src/core/variable.h"

#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> shape;

	shape.push_back(2);
	shape.push_back(2);

	{
		Tensor	input(shape);
		Variable	variable(input, true);

		variable.gradient().data()[0] = 10.0f;
		variable.gradient().data()[1] = 20.0f;

		variable.zero_grad();

		if (!variable.requires_grad())
			return (1);

		if (variable.gradient().data()[0] != 0.0f)
			return (1);

		if (variable.gradient().data()[1] != 0.0f)
			return (1);
	}

	std::cout << "test_variable: OK" << std::endl;
	return (0);
}