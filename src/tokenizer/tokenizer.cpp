#include "tokenizer.h"

#include "../io/json.h"

#include <algorithm>
#include <stdexcept>

Tokenizer::Tokenizer()
	: _unknown_id(0),
		_pad_id(0),
		_bos_id(0),
		_eos_id(0)
{
}

std::vector<std::string>	Tokenizer::split_text(
	const std::string& text) const
{
	std::vector<std::string> words;
	std::string current;
	size_t i;

	i = 0;
	while (i < text.size())
	{
		if (text[i] == ' ')
		{
			if (!current.empty())
			{
				words.push_back(current);
				current.clear();
			}
			words.push_back(" ");
		}
		else
			current += text[i];
		i++;
	}

	if (!current.empty())
		words.push_back(current);

	return (words);
}

std::vector<std::string>	Tokenizer::split_word(
	const std::string& word) const
{
	std::vector<std::string> pieces;
	size_t i;

	i = 0;
	while (i < word.size())
	{
		pieces.push_back(
			std::string(1, word[i]));
		i++;
	}

	return (pieces);
}

std::string	Tokenizer::merge_key(
	const std::string& left,
	const std::string& right) const
{
	return (left + " " + right);
}

std::vector<std::string>	Tokenizer::apply_bpe(
	const std::string& word) const
{
	std::vector<std::string> pieces;
	bool changed;

	pieces = split_word(word);

	if (pieces.size() < 2)
		return (pieces);

	changed = true;
	while (changed)
	{
		size_t best_rank;
		size_t best_position;
		bool found;
		size_t i;

		changed = false;
		best_rank = static_cast<size_t>(-1);
		best_position = 0;
		found = false;

		i = 0;
		while (i + 1 < pieces.size())
		{
			std::string key;

			key = merge_key(pieces[i], pieces[i + 1]);

			std::map<std::string, size_t>::const_iterator
				it = _merge_ranks.find(key);

			if (it != _merge_ranks.end()
				&& (!found || it->second < best_rank))
			{
				best_rank = it->second;
				best_position = i;
				found = true;
			}
			i++;
		}

		if (found)
		{
			std::string merged;

			merged = pieces[best_position]
				+ pieces[best_position + 1];

			pieces[best_position] = merged;

			pieces.erase(
				pieces.begin() + best_position + 1);

			changed = true;
		}
	}

	return (pieces);
}

std::vector<size_t>	Tokenizer::encode(
	const std::string& text) const
{
	std::vector<size_t> tokens;
	std::vector<std::string> words;
	size_t i;

	words = split_text(text);

	i = 0;
	while (i < words.size())
	{
		std::vector<std::string> pieces;
		size_t j;

		if (words[i] == " ")
		{
			std::map<std::string, size_t>::const_iterator it;

			it = _vocab.find(" ");

			if (it != _vocab.end())
				tokens.push_back(it->second);
			else
				tokens.push_back(_unknown_id);

			i++;
			continue ;
		}

		pieces = apply_bpe(words[i]);

		j = 0;
		while (j < pieces.size())
		{
			std::map<std::string, size_t>::const_iterator it;

			it = _vocab.find(pieces[j]);

			if (it != _vocab.end())
				tokens.push_back(it->second);
			else
				tokens.push_back(_unknown_id);

			j++;
		}
		i++;
	}

	return (tokens);
}

std::string	Tokenizer::decode(
	const std::vector<size_t>& tokens) const
{
	std::string text;
	size_t i;

	i = 0;
	while (i < tokens.size())
	{
		std::map<size_t, std::string>::const_iterator it;

		it = _id_to_token.find(tokens[i]);

		if (it != _id_to_token.end())
			text += it->second;
		i++;
	}

	return (text);
}

void	Tokenizer::load(
	const std::string& path)
{
	nlohmann::json data;
	nlohmann::json vocabulary;
	nlohmann::json merges;

	data = Json::load(path);

	vocabulary = data.at("vocab");
	merges = data.at("merges");

	_vocab.clear();
	_id_to_token.clear();
	_merge_ranks.clear();

	for (nlohmann::json::const_iterator it =
		vocabulary.begin(); it != vocabulary.end(); ++it)
	{
		size_t id;

		id = it.value().get<size_t>();

		_vocab[it.key()] = id;
		_id_to_token[id] = it.key();
	}

	size_t i;

	i = 0;
	while (i < merges.size())
	{
		std::string left;
		std::string right;
		std::string key;

		left = merges.at(i).at(0).get<std::string>();
		right = merges.at(i).at(1).get<std::string>();
		key = merge_key(left, right);

		_merge_ranks[key] = i;
		i++;
	}

	_unknown_id =
		data.at("special_tokens").at("unk").get<size_t>();

	_pad_id =
		data.at("special_tokens").at("pad").get<size_t>();

	_bos_id =
		data.at("special_tokens").at("bos").get<size_t>();

	_eos_id =
		data.at("special_tokens").at("eos").get<size_t>();
}

void	Tokenizer::save(
	const std::string& path) const
{
	nlohmann::json data;
	nlohmann::json vocabulary;
	nlohmann::json merges;
	nlohmann::json special_tokens;

	std::map<std::string, size_t>::const_iterator it;

	for (it = _vocab.begin(); it != _vocab.end(); ++it)
		vocabulary[it->first] = it->second;

	std::vector<std::pair<std::string, size_t> > ordered_merges;

	for (it = _merge_ranks.begin(); it != _merge_ranks.end(); ++it)
		ordered_merges.push_back(*it);

	std::sort(
		ordered_merges.begin(),
		ordered_merges.end(),
		[](const std::pair<std::string, size_t>& a,
			const std::pair<std::string, size_t>& b)
		{
			return (a.second < b.second);
		});

	size_t i;

	i = 0;
	while (i < ordered_merges.size())
	{
		nlohmann::json merge;
		std::string key;
		size_t separator;

		key = ordered_merges[i].first;
		separator = key.find(' ');

		merge.push_back(
			key.substr(0, separator));

		merge.push_back(
			key.substr(separator + 1));

		merges.push_back(merge);
		i++;
	}

	special_tokens["unk"] = _unknown_id;
	special_tokens["pad"] = _pad_id;
	special_tokens["bos"] = _bos_id;
	special_tokens["eos"] = _eos_id;

	data["type"] = "bpe";
	data["vocab"] = vocabulary;
	data["merges"] = merges;
	data["special_tokens"] = special_tokens;

	Json::save(path, data);
}

size_t	Tokenizer::vocab_size() const
{
	return (_vocab.size());
}

size_t	Tokenizer::unknown_id() const
{
	return (_unknown_id);
}

size_t	Tokenizer::pad_id() const
{
	return (_pad_id);
}

size_t	Tokenizer::bos_id() const
{
	return (_bos_id);
}

size_t	Tokenizer::eos_id() const
{
	return (_eos_id);
}
