#include "neural_net.hpp"

#include <iostream>
#include <math.h>
#include <random>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cmath>

void NeuralNet::create_net(const std::vector<uint32_t> &topology) {
    srand(time(0));

    for (uint32_t layer = 0; layer < topology.size(); ++layer) {
        uint32_t num_of_neurons = topology[layer];
        _layers.push_back(std::vector<Neuron>());

        for (uint32_t i = 0; i < num_of_neurons; ++i) {
            Neuron n;
            n.a = 1.0;
            n.z = 1.0;

            if (layer != 0) {
                for (uint32_t j = 0; j < _layers[_layers.size() - 2].size(); ++j)
                    n.connections.push_back(((rand() % 2000) - 1000.0) / 1000.0);
            }

            _layers[_layers.size() - 1].push_back(n);
        }
    }
}

void NeuralNet::fire(const std::vector<double> &inputs) {
    if (inputs.size() != _layers[0].size()) {
        std::cout << "Wrong output shape!\n";
        return;
    }

    for (uint32_t i = 0; i < _layers[0].size(); ++i) {
        _layers[0][i].a = inputs[i];
    }


    for (uint32_t i = 1; i < _layers.size(); ++i) {
        for (Neuron &neuron : _layers[i]) {
            double z = 0.0;

            for (uint32_t j = 0; j < neuron.connections.size(); ++j) {
                z += _layers[i - 1][j].a * neuron.connections[j];
            }
            if (i == _layers.size() - 1) {
                neuron.z = z;
                neuron.a = _activation_func(z);
            } else {
                neuron.z = z;
                neuron.a = _activation_func(z);
            }
        }
    }
}

void NeuralNet::get_outputs(std::vector<double> &buffer) {
    buffer.clear();
    for (const Neuron &n : _layers[_layers.size() - 1])
        buffer.push_back(n.a);
}

double NeuralNet::teach(const std::vector<double> &inputs, const std::vector<double> &target, double r) {
    fire(inputs);

    double cost = 0;
    // Output layer
    {
        std::vector<Neuron> &output_layer = _layers[_layers.size() - 1];
        uint32_t i = 0;
        for (Neuron &n : output_layer) {
            cost += pow(n.a - target[i], 2);
            n.gradient = 2 * (n.a - target[i]);
            i += 1;
        }
    }


    // Hidden layers
    for (uint32_t i = _layers.size() - 2; i > 0; --i) {
        std::vector<Neuron> &layer = _layers[i];
        std::vector<Neuron> &next_layer = _layers[i + 1];

        uint32_t n_index = 0;
        for (Neuron &n : layer) {

            double gradient_sum = 0.0;
            for (Neuron &next_n : next_layer) {
                gradient_sum += _activation_func_derivative(next_n.z) * next_n.connections[n_index] * next_n.gradient;
                next_n.connections[n_index] -= n.a * _activation_func_derivative(next_n.z) * next_n.gradient * r;
            }
            n.gradient = gradient_sum;

            n_index += 1;
        }
    }

    return cost;
}

double NeuralNet::_activation_func(double val) {
    return tanh(val);
}

double NeuralNet::_activation_func_derivative(double val) {
    double th = tanh(val);
    return 1.0 - th * th;
}
