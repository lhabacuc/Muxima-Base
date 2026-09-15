#include "dataset.h"

#include <fstream>
#include <stdexcept>

Dataset::Dataset(
	const std::string& corpus_path,
	const std::string& tokenizer_path,
	size_t sequence_length)
	: _sequence_length(sequence_length)
{
	if (sequence_length < 2)
		throw std::invalid_argument(
			"Sequence length must be at least 2");

	_tokenizer.load(tokenizer_path);
	load_tokens(corpus_path);
}

void	Dataset::load_tokens(
	const std::string& path)
{
	std::ifstream file(path.c_str());
	std::string line;

	if (!file.is_open())
		throw std::runtime_error(
			"Could not open corpus: " + path);

	while (std::getline(file, line))
		add_text(line);

	file.close();
}

void	Dataset::add_text(
	const std::string& text)
{
	std::vector<size_t> tokens;

	tokens = _tokenizer.encode(text);

	_tokens.insert(
		_tokens.end(),
		tokens.begin(),
		tokens.end());
}

TrainingSample	Dataset::get_sample(
	size_t index) const
{
	TrainingSample sample;
	size_t i;

	if (index >= size())
		throw std::out_of_range(
			"Dataset sample index out of range");

	i = 0;
	while (i < _sequence_length)
	{
		sample.input.push_back(
			_tokens[index + i]);

		sample.target.push_back(
			_tokens[index + i + 1]);

		i++;
	}

	return (sample);
}

size_t	Dataset::size() const
{
	if (_tokens.size() <= _sequence_length)
		return (0);

	return (_tokens.size() - _sequence_length);
}

size_t	Dataset::sequence_length() const
{
	return (_sequence_length);
}

const std::vector<size_t>&	Dataset::tokens() const
{
	return (_tokens);
}
