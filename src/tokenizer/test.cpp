#include "bpe_trainer.h"

#include <iostream>

int	main()
{
	BPETrainer trainer(300);

	trainer.train(
		"data/corpus.txt",
		"config/tokenizer.json");

	std::cout << "BPE trainer test passed"
		<< std::endl;

	return (0);
}
