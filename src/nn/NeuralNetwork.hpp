#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <cuLA.cuh>

#include <vector>
#include <type_traits>

template<typename T>
using vector = std::vector<T>;

template<typename Expected, typename... Ts>
inline constexpr bool are_all_same_as_v = std::conjunction_v<std::is_same<Expected, Ts>...>;

class NeuralNetwork {
public:
	vector<Matrix> Weights;
	vector<Vector> Biases;
	vector<Vector> activations;
	vector<Vector> zs;

	template<typename... Args>
	NeuralNetwork(Args... args);

	Vector forward(const Vector& input);

	void backprop(const Vector& input, const Vector& target, float learningRate = 0.1f);
};

#endif // NEURALNETWORK_HPP