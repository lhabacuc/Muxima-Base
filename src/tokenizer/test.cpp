#include "tokenizer.h"

#include <cassert>
#include <iostream>
#include <vector>

int	main()
{
	Tokenizer tokenizer;
	std::vector<size_t> tokens;
	std::string decoded;

	tokenizer.load("config/tokenizer.json");

	assert(tokenizer.vocab_size() == 21);
	assert(tokenizer.unknown_id() == 0);
	assert(tokenizer.pad_id() == 1);
	assert(tokenizer.bos_id() == 2);
	assert(tokenizer.eos_id() == 3);

	tokens = tokenizer.encode("ab");

	assert(tokens.size() == 1);
	assert(tokens[0] == 18);

	decoded = tokenizer.decode(tokens);

	assert(decoded == "ab");

	std::cout << "Tokenizer tests passed"
		<< std::endl;

	return (0);
}
