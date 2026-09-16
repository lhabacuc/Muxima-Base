#ifndef CONFIG_H
# define CONFIG_H

# include <cstddef>
# include <string>

struct ModelConfig
{
	std::string	name;
	size_t	vocab_size;
	size_t	max_length;
	size_t	embedding_dim;
	size_t	num_layers;
	size_t	num_heads;
	size_t	hidden_dim;
};

struct TrainingConfig
{
	float	learning_rate;
	size_t	batch_size;
	size_t	epochs;
};

struct TokenizerConfig
{
	std::string	type;
	size_t		vocab_size;
};

class Config
{
	public:
		ModelConfig		model;
		TrainingConfig	training;
		TokenizerConfig	tokenizer;

		static Config	load(
			const std::string& path);
};

#endif
