#include "../src/training/cross_entropy.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int	main()
{
	std::vector<size_t>	logit_shape;
	std::vector<size_t>	target_shape;

	logit_shape.push_back(2);
	logit_shape.push_back(3);

	target_shape.push_back(2);

	Tensor logits(logit_shape);
	Tensor targets(target_shape);

	logits.data()[0] = 2.0f;
	logits.data()[1] = 1.0f;
	logits.data()[2] = 0.0f;

	logits.data()[3] = 0.0f;
	logits.data()[4] = 2.0f;
	logits.data()[5] = 0.0f;

	targets.data()[0] = 0.0f;
	targets.data()[1] = 1.0f;

	float loss = CrossEntropy::forward(
		logits,
		targets);

	Tensor gradient = CrossEntropy::backward(
		logits,
		targets);

	assert(loss > 0.0f);
	assert(gradient.shape()[0] == 2);
	assert(gradient.shape()[1] == 3);

	assert(gradient.data()[0] < 0.0f);
	assert(gradient.data()[4] < 0.0f);

	std::cout << "CrossEntropy tests passed"
		<< std::endl;

	return (0);
}
