#pragma once

#include "../neuron/neuron.hpp"

#include <vector>
#include <cstdint>

class NeuralNet {
    private:
        uint32_t _batch_size;
        uint32_t _training_counter;

        std::vector<std::vector<Neuron>> _layers;

        double _activation_func(double val);

        double _activation_func_derivative(double val);

    public:
        void create_net(const std::vector<uint32_t> &topology);

        void fire(const std::vector<double> &inputs);

        void get_outputs(std::vector<double> &buffer);

        double teach(const std::vector<double> &inputs, const std::vector<double> &target, double r);

        void set_batch_size(uint32_t size);

        void get_connections(std::vector<double> &inputs);

};
