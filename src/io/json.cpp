#include "json.h"

#include <fstream>
#include <stdexcept>

nlohmann::json	Json::load(
	const std::string& path)
{
	std::ifstream file(path.c_str());
	nlohmann::json data;

	if (!file.is_open())
		throw std::runtime_error(
			"Could not open JSON file: " + path);

	try
	{
		file >> data;
	}
	catch (const nlohmann::json::parse_error& error)
	{
		throw std::runtime_error(
			"Invalid JSON file: " + path);
	}

	file.close();
	return (data);
}

void	Json::save(
	const std::string& path,
	const nlohmann::json& data)
{
	std::ofstream file(path.c_str());

	if (!file.is_open())
		throw std::runtime_error(
			"Could not create JSON file: " + path);

	file << data.dump(4);
	file << std::endl;

	file.close();
}
