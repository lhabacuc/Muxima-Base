#include "checkpoint.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static Tensor	create_tensor()
{
	std::vector<size_t> shape;

	shape.push_back(2);
	shape.push_back(3);

	Tensor tensor(shape);

	tensor.data()[0] = 1.0f;
	tensor.data()[1] = 2.0f;
	tensor.data()[2] = 3.0f;
	tensor.data()[3] = 4.0f;
	tensor.data()[4] = 5.0f;
	tensor.data()[5] = 6.0f;

	return (tensor);
}

int	main()
{
	Tensor original = create_tensor();
	Checkpoint checkpoint;
	CheckpointParameter parameter;

	checkpoint.model_name = "Muxima-Base";
	checkpoint.step = 100;
	checkpoint.epoch = 2;
	checkpoint.loss = 1.25f;

	parameter = CheckpointIO::from_tensor(
		"test.weight",
		original);

	checkpoint.parameters.push_back(parameter);

	CheckpointIO::save(
		"checkpoint_test.json",
		checkpoint);

	Checkpoint loaded =
		CheckpointIO::load(
			"checkpoint_test.json");

	assert(loaded.model_name == "Muxima-Base");
	assert(loaded.step == 100);
	assert(loaded.epoch == 2);
	assert(std::fabs(
		loaded.loss - 1.25f) < 0.0001f);

	assert(loaded.parameters.size() == 1);
	assert(loaded.parameters[0].name ==
		"test.weight");

	Tensor restored = CheckpointIO::to_tensor(
		loaded.parameters[0]);

	assert(restored.shape().size() == 2);
	assert(restored.shape()[0] == 2);
	assert(restored.shape()[1] == 3);
	assert(restored.data()[0] == 1.0f);
	assert(restored.data()[5] == 6.0f);

	std::cout << "Checkpoint tests passed"
		<< std::endl;

	return (0);
}
