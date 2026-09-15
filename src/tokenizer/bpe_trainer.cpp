#include "bpe_trainer.h"

#include "../io/json.h"

#include <fstream>
#include <stdexcept>
#include <sstream>

BPETrainer::BPETrainer(size_t vocab_size)
	: _vocab_size(vocab_size),
		_next_id(0)
{
	if (vocab_size < 256)
		throw std::invalid_argument(
			"Vocabulary size must be at least 256");
}

std::string	BPETrainer::merge_key(
	const std::string& left,
	const std::string& right) const
{
	return (left + " " + right);
}

void	BPETrainer::initialize_vocab(
	const std::vector<std::string>& words)
{
	size_t i;

	i = 0;
	while (i < 256)
	{
		std::string token;

		token += static_cast<char>(i);
		_vocab[token] = _next_id;
		_next_id++;
		i++;
	}

	_vocab["<unk>"] = _next_id++;
	_vocab["<pad>"] = _next_id++;
	_vocab["<bos>"] = _next_id++;
	_vocab["<eos>"] = _next_id++;

	(void)words;
}

std::map<std::string, size_t>	BPETrainer::count_pairs(
	const std::vector<std::vector<std::string> >& words) const
{
	std::map<std::string, size_t> counts;
	size_t i;

	i = 0;
	while (i < words.size())
	{
		size_t j;

		j = 0;
		while (j + 1 < words[i].size())
		{
			std::string key;

			key = merge_key(
				words[i][j],
				words[i][j + 1]);

			counts[key]++;
			j++;
		}
		i++;
	}

	return (counts);
}

void	BPETrainer::merge_pair(
	std::vector<std::string>& word,
	const std::string& left,
	const std::string& right)
{
	std::vector<std::string> result;
	size_t i;

	i = 0;
	while (i < word.size())
	{
		if (i + 1 < word.size()
			&& word[i] == left
			&& word[i + 1] == right)
		{
			result.push_back(left + right);
			i += 2;
		}
		else
		{
			result.push_back(word[i]);
			i++;
		}
	}

	word = result;
}

void	BPETrainer::train(
	const std::string& input_path,
	const std::string& output_path)
{
	std::ifstream file(input_path.c_str());
	std::vector<std::vector<std::string> > words;

	if (!file.is_open())
		throw std::runtime_error(
			"Could not open training corpus");

	std::string line;

	while (std::getline(file, line))
	{
		size_t i;

		i = 0;
		while (i < line.size())
		{
			std::vector<std::string> word;

			while (i < line.size()
				&& line[i] != ' ')
			{
				std::string byte;

				byte += line[i];
				word.push_back(byte);
				i++;
			}

			if (!word.empty())
				words.push_back(word);

			while (i < line.size()
				&& line[i] == ' ')
				i++;
		}
	}

	file.close();

	initialize_vocab(std::vector<std::string>());

	while (_next_id < _vocab_size)
	{
		std::map<std::string, size_t> counts;
		std::map<std::string, size_t>::const_iterator best;
		std::string left;
		std::string right;
		size_t separator;
		size_t i;

		counts = count_pairs(words);

		if (counts.empty())
			break;

		best = counts.begin();

		for (std::map<std::string, size_t>::const_iterator it =
			counts.begin(); it != counts.end(); ++it)
		{
			if (it->second > best->second)
				best = it;
		}

		separator = best->first.find(' ');

		left = best->first.substr(0, separator);
		right = best->first.substr(separator + 1);

		_vocab[left + right] = _next_id++;
		_merges.push_back(
			std::make_pair(left, right));

		i = 0;
		while (i < words.size())
		{
			merge_pair(words[i], left, right);
			i++;
		}
	}

	nlohmann::json result;
	nlohmann::json vocabulary;
	nlohmann::json merges;
	nlohmann::json special_tokens;

	for (std::map<std::string, size_t>::const_iterator it =
		_vocab.begin(); it != _vocab.end(); ++it)
	{
		vocabulary[it->first] = it->second;
	}

	size_t i;

	i = 0;
	while (i < _merges.size())
	{
		nlohmann::json pair;

		pair.push_back(_merges[i].first);
		pair.push_back(_merges[i].second);

		merges.push_back(pair);
		i++;
	}

	special_tokens["unk"] = _vocab["<unk>"];
	special_tokens["pad"] = _vocab["<pad>"];
	special_tokens["bos"] = _vocab["<bos>"];
	special_tokens["eos"] = _vocab["<eos>"];

	result["type"] = "bpe";
	result["vocab"] = vocabulary;
	result["merges"] = merges;
	result["special_tokens"] = special_tokens;

	Json::save(output_path, result);
}
