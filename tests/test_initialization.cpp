#include "../src/nn/initialization.h"

#include <iostream>

int	main()
{
	Tensor	weights = xavier_uniform(128, 64);

	if (weights.shape()[0] != 128)
		return (1);
	if (weights.shape()[1] != 64)
		return (1);
	if (weights.size() != 128 * 64)
		return (1);

	std::cout << "test_xavier: OK" << std::endl;
	return (0);
}
