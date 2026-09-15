#include "config.h"
#include "json.h"

Config	Config::load(
	const std::string& path)
{
	nlohmann::json data;
	Config config;

	data = Json::load(path);

	config.model.vocab_size =
		data.at("model").at("vocab_size");

	config.model.max_length =
		data.at("model").at("max_length");

	config.model.embedding_dim =
		data.at("model").at("embedding_dim");

	config.model.num_layers =
		data.at("model").at("num_layers");

	config.model.num_heads =
		data.at("model").at("num_heads");

	config.model.hidden_dim =
		data.at("model").at("hidden_dim");

	config.training.learning_rate =
		data.at("training").at("learning_rate");

	config.training.batch_size =
		data.at("training").at("batch_size");

	config.training.epochs =
		data.at("training").at("epochs");

	config.tokenizer.type =
		data.at("tokenizer").at("type");

	config.tokenizer.vocab_size =
		data.at("tokenizer").at("vocab_size");

	return (config);
}
