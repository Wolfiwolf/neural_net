#include "neural_net/neural_net.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>

void generate_training_set(
        uint32_t size, 
        std::vector<std::vector<double>> &training_inputs, 
        std::vector<std::vector<double>> &target_outputs) {
    training_inputs.clear();
    target_outputs.clear();

    std::cout << "Generating training data...\n";
    for (uint32_t i = 0; i < size; ++i) {
        uint8_t a = rand() % 128;
        uint8_t b = rand() % 128;
        uint8_t c = a + b;

        std::vector<double> input_bits;
        std::vector<double> target_bits;
        for (uint8_t j = 0; j < 7; j++)
            input_bits.push_back((a & (1 << j)) ? 1.0 : 0.0);
        for (uint8_t j = 0; j < 7; j++)
            input_bits.push_back((b & (1 << j)) ? 1.0 : 0.0);
        for (uint8_t j = 0; j < 8; j++) {
            target_bits.push_back(c & (1 << j) ? 1.0 : 0.0);
        }
        training_inputs.push_back(input_bits);
        target_outputs.push_back(target_bits);
        printf("%.2f%%   \r", (i / (double)size) * 100.0);
    }
    printf("%.2f%%\n", 100.0);

}

void teach_addition(NeuralNet &net) {
    uint32_t N = 2000000;
    net.create_net({14, 32, 32, 32, 8});
    net.set_batch_size(1);
    double learning_rate = 0.0001;

    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> target_outputs;
    generate_training_set(N, training_inputs, target_outputs);

    std::ofstream cost_values("learning_costs.txt");
    for (uint32_t epoch = 0; epoch < 1; ++epoch) {
        std::cout << "Epoch: " << epoch + 1 << "\n";
        double cost_average = 0;
        double prev_cost_average = 0;
        std::cout << "Training...\n";
        for (uint32_t i = 0; i < N; ++i) {
            double cost = net.teach(training_inputs[i], target_outputs[i], learning_rate);

            cost_average += cost;
            if (i % 1000 == 0) {
                cost_average = cost_average / 1000.0;
                cost_values << cost_average << "\n";
                prev_cost_average = cost_average;
                cost_average = 0;
                cost_values.flush();
            }

            printf("%.2f%% (average cost: %.5f)   \r", (i / (double)N) * 100.0, prev_cost_average);
        }
        printf("%.2f%% (average cost: %.5f)   \n", 100.0, prev_cost_average);

    }
    cost_values.close();


}

int add_with_net(NeuralNet &net, uint8_t a, uint8_t b) {
    std::vector<double> input_bits;
    for (uint8_t j = 0; j < 7; j++)
        input_bits.push_back(a & (1 << j) ? 1.0 : 0.0);
    for (uint8_t j = 0; j < 7; j++)
        input_bits.push_back(b & (1 << j) ? 1.0 : 0.0);

    net.fire(input_bits);

    std::vector<double> output_bits;
    net.get_outputs(output_bits);

    uint8_t result = 0;
    for (uint32_t i = 0; i < output_bits.size(); ++i) {
        if (output_bits[i] > 0.5) {
            result |= (1 << i);
        }
    }

    return result;
}

int main() {

    NeuralNet net;
    teach_addition(net);

    uint16_t num_of_tests = 1000;
    uint8_t correct_count = 0;
    for (uint16_t i = 0; i < num_of_tests; ++i) {
        uint8_t a = rand() % 128;
        uint8_t b = rand() % 128;
        uint8_t c = a + b;

        uint8_t result = add_with_net(net, a, b);
        printf("%d + %d = %d\t%s\n", a, b, result, a + b == result ? "True" : "False");

        if (a + b == result) correct_count += 1;
    }

    printf("Success rate is %.2f%%\n", (float)correct_count / num_of_tests);


    return 0;
}
