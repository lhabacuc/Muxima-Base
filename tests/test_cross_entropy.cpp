#include "../src/training/cross_entropy.h"

#include <cmath>
#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t> logits_shape;
	std::vector<size_t> target_shape;

	logits_shape.push_back(2);
	logits_shape.push_back(3);

	target_shape.push_back(2);

	{
		Tensor	logits(logits_shape);
		Tensor	targets(target_shape);
		float	loss;

		logits.data()[0] = 0.0f;
		logits.data()[1] = 0.0f;
		logits.data()[2] = 5.0f;

		logits.data()[3] = 5.0f;
		logits.data()[4] = 0.0f;
		logits.data()[5] = 0.0f;

		targets.data()[0] = 2.0f;
		targets.data()[1] = 0.0f;

		loss = CrossEntropy::forward(logits, targets);

		if (loss <= 0.0f)
			return (1);

		if (loss > 0.1f)
			return (1);
	}

	std::cout << "test_cross_entropy: OK"
		<< std::endl;
	return (0);
}