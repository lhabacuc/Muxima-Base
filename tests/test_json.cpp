#include "../src/io/json.h"
#include "../src/io/config.h"

#include <cassert>
#include <iostream>

int	main()
{
	Config config;

	config = Config::load("config/model.json");

	assert(config.model.vocab_size == 32000);
	assert(config.model.max_length == 1024);
	assert(config.model.embedding_dim == 512);
	assert(config.model.num_layers == 8);
	assert(config.model.num_heads == 8);
	assert(config.model.hidden_dim == 2048);

	assert(config.training.learning_rate == 0.0003f);
	assert(config.training.batch_size == 8);
	assert(config.training.epochs == 3);

	assert(config.tokenizer.type == "bpe");
	assert(config.tokenizer.vocab_size == 32000);

	std::cout << "JSON tests passed"
		<< std::endl;

	return (0);
}
