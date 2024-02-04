#pragma once

#include <vector>

class Neuron {
    public:
        double gradient;
        double z;
        double a;
        std::vector<double> connections;
};
