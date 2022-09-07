/**
 * @file main.c
 * @author Auracle
 * @brief This is an example of a XOR neural network using a single hidden layer with 2 neurons.
 * @version 1.0
 * @date 2022-09-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Seed - Set to 0 for random seed
#define SEED 694201337
#define WEIGHTS 2
#define HIDDEN 3
#define OUTPUTS 1
#define LEARNING_RATE 2.009
#define TRAINING_SET_SIZE 4
#define EPOCHS 100000

typedef struct
{
    double weights[WEIGHTS];
    double bias;
} Neuron;

typedef struct
{
    Neuron neurons[HIDDEN];
} Layer;

typedef struct
{
    Layer hidden;
    Neuron output;
} Network;

typedef struct
{
    double inputs[WEIGHTS];
    double expected;
} TrainingSet;

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x)
{
    return x * (1 - x);
}

void networkInitialize(Network *network)
{
    // For each neuron in the hidden layer
    for (int i = 0; i < HIDDEN; i++)
    {
        // For each weight in the neuron
        for (int j = 0; j < WEIGHTS; j++)
        {
            // Initialize weights to random values between -1 and 1
            network->hidden.neurons[i].weights[j] = (double)rand() / RAND_MAX;
        }
        // Initialize bias to random value between -1 and 1
        network->hidden.neurons[i].bias = (double)rand() / RAND_MAX;
    }

    // For each neuron in the output layer
    for (int i = 0; i < OUTPUTS; i++)
    {
        // For each weight in the neuron
        for (int j = 0; j < WEIGHTS; j++)
        {
            // Initialize weights to random values between -1 and 1
            network->output.weights[j] = (double)rand() / RAND_MAX;
        }
        // Initialize bias to random value between -1 and 1
        network->output.bias = (double)rand() / RAND_MAX;
    }
}

double feedForward(Network *network, double inputs[WEIGHTS])
{
    // Outputs of the hidden layer
    double hiddenOutputs[HIDDEN];
    // For each neuron in the hidden layer
    for (int i = 0; i < HIDDEN; i++)
    {
        // Sum of the inputs * weights + bias
        double sum = 0;
        for (int j = 0; j < WEIGHTS; j++)
        {
            // Add the product of the input and the weight to the sum
            sum += inputs[j] * network->hidden.neurons[i].weights[j];
        }
        // Add the bias to the sum
        sum += network->hidden.neurons[i].bias;
        // Set the output of the neuron to the sigmoid of the sum
        hiddenOutputs[i] = sigmoid(sum);
    }

    // Output of the output neuron
    double output = 0;
    // For each neuron in the output layer
    for (int i = 0; i < HIDDEN; i++)
    {
        // Sum of the inputs * weights + bias
        output += hiddenOutputs[i] * network->output.weights[i];
    }
    // Add the bias to the sum
    output += network->output.bias;
    // Set the output of the neuron to the sigmoid of the sum
    output = sigmoid(output);

    return output;
}

void train(Network *network, TrainingSet trainingSet[TRAINING_SET_SIZE])
{
    // For each epoch
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        // For each training set
        for (int i = 0; i < TRAINING_SET_SIZE; i++)
        {
            // Inputs of the network
            double inputs[WEIGHTS];
            // For each input
            for (int j = 0; j < WEIGHTS; j++)
            {
                // Set the input to the training set input
                inputs[j] = trainingSet[i].inputs[j];
            }

            // Outputs of the hidden layer
            double hiddenOutputs[HIDDEN];
            // For each neuron in the hidden layer
            for (int j = 0; j < HIDDEN; j++)
            {
                // Sum of the inputs * weights + bias
                double sum = 0;
                // For each weight in the neuron
                for (int k = 0; k < WEIGHTS; k++)
                {
                    // Add the product of the input and the weight to the sum
                    sum += inputs[k] * network->hidden.neurons[j].weights[k];
                }
                // Add the bias to the sum
                sum += network->hidden.neurons[j].bias;
                // Set the output of the neuron to the sigmoid of the sum
                hiddenOutputs[j] = sigmoid(sum);
            }

            // Output of the output neuron
            double output = 0;
            // For each neuron in the output layer
            for (int j = 0; j < HIDDEN; j++)
            {
                // Sum of the inputs * weights + bias
                output += hiddenOutputs[j] * network->output.weights[j];
            }
            // Add the bias to the sum
            output += network->output.bias;
            // Set the output of the neuron to the sigmoid of the sum
            output = sigmoid(output);

            // Error of the output neuron
            double error = trainingSet[i].expected - output;

            // Derivative of the sigmoid of the output
            double outputGradient = sigmoidDerivative(output);
            // Delta of the output neuron
            double outputDelta = error * outputGradient;

            // Derivative of the sigmoid of the hidden layer
            double hiddenGradients[HIDDEN];
            // For each neuron in the hidden layer
            for (int j = 0; j < HIDDEN; j++)
            {
                // Set the derivative of the sigmoid of the output to the derivative of the sigmoid of the hidden layer
                hiddenGradients[j] = sigmoidDerivative(hiddenOutputs[j]);
            }

            // Deltas of the hidden layer
            double hiddenDeltas[HIDDEN];
            // For each neuron in the hidden layer
            for (int j = 0; j < HIDDEN; j++)
            {
                // Set the delta of the hidden layer to the delta of the output neuron * the weight of the output neuron
                hiddenDeltas[j] = outputDelta * network->output.weights[j] * hiddenGradients[j];
            }

            // For each neuron in the hidden layer
            for (int j = 0; j < HIDDEN; j++)
            {
                // For each weight in the neuron
                for (int k = 0; k < WEIGHTS; k++)
                {
                    // Adjust the weight by the learning rate * the delta of the hidden layer * the input of the hidden layer
                    network->hidden.neurons[j].weights[k] += LEARNING_RATE * hiddenDeltas[j] * inputs[k];
                }
                // Adjust the bias by the learning rate * the delta of the hidden layer
                network->hidden.neurons[j].bias += LEARNING_RATE * hiddenDeltas[j];
            }

            // For each weight in the neuron
            for (int j = 0; j < HIDDEN; j++)
            {
                // Adjust the weight by the learning rate * the delta of the output neuron * the output of the hidden layer
                network->output.weights[j] += LEARNING_RATE * outputDelta * hiddenOutputs[j];
            }
            // Adjust the bias by the learning rate * the delta of the output neuron
            network->output.bias += LEARNING_RATE * outputDelta;
        }
    }
}

int main()
{
    // Seed the random number generator
    if (SEED == 0)
    {
        srand(SEED);
    }
    else
    {
        srand(time(NULL));
    }

    // Initialize the network
    Network network;
    // Initialize the training set
    networkInitialize(&network);

    // Initialize the training set
    TrainingSet trainingSet[TRAINING_SET_SIZE];
    // For each training set
    for (int i = 0; i < TRAINING_SET_SIZE; i++)
    {
        // For each input
        for (int j = 0; j < WEIGHTS; j++)
        {
            // Set the input to a random value between 0 and 1
            trainingSet[i].inputs[j] = (double)rand() / RAND_MAX;
        }
        // Set the expected output to the XOR of the inputs
        trainingSet[i].expected = trainingSet[i].inputs[0] != trainingSet[i].inputs[1];
    }

    // Train the network
    train(&network, trainingSet);

    // For each training set
    for (int i = 0; i < TRAINING_SET_SIZE; i++)
    {
        // Print the inputs
        printf("%f XOR %f = %f (expected %f)\r\n", trainingSet[i].inputs[0], trainingSet[i].inputs[1], feedForward(&network, trainingSet[i].inputs), trainingSet[i].expected);
    }

    // Successfull exit
    return 0;
}