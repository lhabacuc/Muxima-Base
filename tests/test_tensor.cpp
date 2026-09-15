#include "../src/core/tensor.h"

#include <iostream>
#include <vector>

static int	test_shape()
{
	std::vector<size_t> shape;
	Tensor tensor(shape);

	if (tensor.shape().size() != 0)
		return (1);
	return (0);
}

static int	test_matrix()
{
	std::vector<size_t> shape;
	Tensor tensor(shape);

	shape.push_back(2);
	shape.push_back(3);
	tensor = Tensor(shape);

	if (tensor.shape().size() != 2)
		return (1);
	if (tensor.shape()[0] != 2)
		return (1);
	if (tensor.shape()[1] != 3)
		return (1);
	if (tensor.size() != 6)
		return (1);
	return (0);
}

int	main()
{
	if (test_shape() != 0)
	{
		std::cout << "test_shape: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_shape: OK" << std::endl;

	if (test_matrix() != 0)
	{
		std::cout << "test_matrix: FAIL" << std::endl;
		return (1);
	}
	std::cout << "test_matrix: OK" << std::endl;

	return (0);
}
