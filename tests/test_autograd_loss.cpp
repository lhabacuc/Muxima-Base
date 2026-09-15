#include "../src/core/autograd_graph.h"

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

	Tensor		logits_value(logit_shape);
	Tensor		targets(target_shape);
	Variable	logits(logits_value, true);
	AutogradGraph	graph;
	Variable	*loss;

	logits.value().data()[0] = 2.0f;
	logits.value().data()[1] = 1.0f;
	logits.value().data()[2] = 0.0f;

	logits.value().data()[3] = 0.0f;
	logits.value().data()[4] = 2.0f;
	logits.value().data()[5] = 0.0f;

	targets.data()[0] = 0.0f;
	targets.data()[1] = 1.0f;

	loss = graph.cross_entropy(
		logits,
		targets);

	assert(loss->value().data()[0] > 0.0f);

	graph.backward(*loss);

	assert(logits.gradient().shape()[0] == 2);
	assert(logits.gradient().shape()[1] == 3);

	assert(logits.gradient().data()[0] < 0.0f);
	assert(logits.gradient().data()[4] < 0.0f);

	std::cout << "Autograd loss tests passed"
		<< std::endl;

	return (0);
}
