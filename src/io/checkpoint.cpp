#include "checkpoint.h"
#include "json.h"

#include <stdexcept>

static nlohmann::json	vector_to_json(
	const std::vector<float>& values)
{
	nlohmann::json result = nlohmann::json::array();

	size_t i;

	i = 0;
	while (i < values.size())
	{
		result.push_back(values[i]);
		i++;
	}
	return (result);
}

static std::vector<float>	json_to_vector(
	const nlohmann::json& data)
{
	std::vector<float> result;
	size_t i;

	if (!data.is_array())
		throw std::runtime_error(
			"Checkpoint data must be an array");

	i = 0;
	while (i < data.size())
	{
		result.push_back(data.at(i).get<float>());
		i++;
	}
	return (result);
}

static nlohmann::json	shape_to_json(
	const std::vector<size_t>& shape)
{
	nlohmann::json result = nlohmann::json::array();

	size_t i;

	i = 0;
	while (i < shape.size())
	{
		result.push_back(shape[i]);
		i++;
	}
	return (result);
}

static std::vector<size_t>	json_to_shape(
	const nlohmann::json& data)
{
	std::vector<size_t> result;
	size_t i;

	if (!data.is_array())
		throw std::runtime_error(
			"Checkpoint shape must be an array");

	i = 0;
	while (i < data.size())
	{
		result.push_back(
			data.at(i).get<size_t>());
		i++;
	}
	return (result);
}

void	CheckpointIO::save(
	const std::string& path,
	const Checkpoint& checkpoint)
{
	nlohmann::json data;
	nlohmann::json parameters =
		nlohmann::json::array();

	size_t i;

	data["model_name"] = checkpoint.model_name;
	data["step"] = checkpoint.step;
	data["epoch"] = checkpoint.epoch;
	data["loss"] = checkpoint.loss;

	i = 0;
	while (i < checkpoint.parameters.size())
	{
		nlohmann::json parameter;

		parameter["name"] =
			checkpoint.parameters[i].name;

		parameter["shape"] =
			shape_to_json(
				checkpoint.parameters[i].shape);

		parameter["data"] =
			vector_to_json(
				checkpoint.parameters[i].data);

		parameters.push_back(parameter);
		i++;
	}

	data["parameters"] = parameters;

	Json::save(path, data);
}

Checkpoint	CheckpointIO::load(
	const std::string& path)
{
	nlohmann::json data;
	Checkpoint checkpoint;

	data = Json::load(path);

	checkpoint.model_name =
		data.at("model_name").get<std::string>();

	checkpoint.step =
		data.at("step").get<size_t>();

	checkpoint.epoch =
		data.at("epoch").get<size_t>();

	checkpoint.loss =
		data.at("loss").get<float>();

	const nlohmann::json& parameters =
		data.at("parameters");

	if (!parameters.is_array())
		throw std::runtime_error(
			"Checkpoint parameters must be an array");

	size_t i;

	i = 0;
	while (i < parameters.size())
	{
		CheckpointParameter parameter;
		const nlohmann::json& item =
			parameters.at(i);

		parameter.name =
			item.at("name").get<std::string>();

		parameter.shape =
			json_to_shape(item.at("shape"));

		parameter.data =
			json_to_vector(item.at("data"));

		checkpoint.parameters.push_back(parameter);
		i++;
	}

	return (checkpoint);
}

CheckpointParameter	CheckpointIO::from_tensor(
	const std::string& name,
	const Tensor& tensor)
{
	CheckpointParameter parameter;

	parameter.name = name;
	parameter.shape = tensor.shape();
	parameter.data = tensor.data();

	return (parameter);
}

Tensor	CheckpointIO::to_tensor(
	const CheckpointParameter& parameter)
{
	Tensor tensor(parameter.shape);

	if (tensor.size() != parameter.data.size())
		throw std::runtime_error(
			"Checkpoint tensor size mismatch");

	size_t i;

	i = 0;
	while (i < tensor.size())
	{
		tensor.data()[i] =
			parameter.data[i];
		i++;
	}

	return (tensor);
}
