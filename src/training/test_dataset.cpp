#include "dataset.h"

#include <cassert>
#include <iostream>

int	main()
{
	Dataset dataset(
		"data/corpus.txt",
		"config/tokenizer.json",
		4);

	assert(dataset.sequence_length() == 4);
	assert(dataset.tokens().size() > 4);
	assert(dataset.size() > 0);

	TrainingSample sample;

	sample = dataset.get_sample(0);

	assert(sample.input.size() == 4);
	assert(sample.target.size() == 4);

	assert(
		sample.target[0] ==
		sample.input[1]);

	assert(
		sample.target[1] ==
		sample.input[2]);

	std::cout << "Dataset tests passed"
		<< std::endl;

	return (0);
}
