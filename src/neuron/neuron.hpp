#pragma once

#include <vector>

class Neuron {
    public:
        bool is_bias;
        double gradient;
        double average_gradient;
        double z;
        double a;
        std::vector<double> connections;
};
