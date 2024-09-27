#ifndef NEURAL_NET_HPP

#include <iomanip>
#include <iostream>
#include <cmath>

#include "random.hpp"

struct Matrix
{
    int rows = 0;
    int cols = 0;
    int numValues = 0;
    float *values;

    Matrix(int _rows, int _cols)
    {
        rows = _rows;
        cols = _cols;
        numValues = rows * cols;
        values = new float[numValues];
        zeros();
    }

    // This starts making problems when the bezier curves are in a std::vector
    ~Matrix()
    {
        delete[] values;
    }

    void mul(const float val) {
        for (int i = 0; i < numValues; i++)
        {
            values[i] *= val;
        }
    }

    void add(const Matrix &other) {
        for (int i = 0; i < numValues; i++)
        {
            values[i] += other.values[i];
        }
    }

    void addRand(uint32_t &randSeed, const float std)
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] += randDist(0.0f, std, randSeed);
        }
    }

    void zeros()
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] = 0.0f;
        }
    }

    void copy(Matrix &other)
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] = other.values[i];
        }
    }

    void print(std::string name)
    {
        std::cout << name << std::endl;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                std::cout << values[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // New softmax function
    void softmax()
    {
        for (int i = 0; i < rows; i++)
        {
            float max_val = values[i * cols];
            float sum = 0.0f;

            // Find max value in the row
            for (int j = 1; j < cols; j++)
            {
                if (values[i * cols + j] > max_val)
                {
                    max_val = values[i * cols + j];
                }
            }

            // Compute exp and sum
            for (int j = 0; j < cols; j++)
            {
                values[i * cols + j] = std::exp(values[i * cols + j] - max_val);
                sum += values[i * cols + j];
            }

            // Normalize
            for (int j = 0; j < cols; j++)
            {
                values[i * cols + j] /= sum;
            }
        }
    }
};

/*
Snake Model:

INPUTS:
weight0 (size * size, hiddenSize)
weight1 (size * size, hiddenSize)
weight2 (hiddenSize, 3)
nParams = 2 * size * size * hiddenSize + hiddenSize * 3

out = activation(board @ weight0 + weight1[applePos]) @ weight2
*/

struct SnakeModel
{
    Matrix weight0;
    Matrix weight1;
    Matrix weight2;

    Matrix hidden;

    int size;
    int hiddenSize;

    SnakeModel(int _size, int _hiddenSize)
        : weight0(_size * _size, _hiddenSize),
          weight1(_size * _size, _hiddenSize),
          weight2(_hiddenSize, 3),
          hidden(1, _hiddenSize)
    {
        size = _size;
        hiddenSize = _hiddenSize;
    }

    int getNumParams()
    {
        return weight0.numValues + weight1.numValues + weight2.numValues;
    }

    void copyWeights(SnakeModel &other)
    {
        weight0.copy(other.weight0);
        weight1.copy(other.weight1);
        weight2.copy(other.weight2);
    }

    void addRand(uint32_t &randSeed, const float std)
    {
        weight0.addRand(randSeed, std);
        weight1.addRand(randSeed, std);
        weight2.addRand(randSeed, std);
    }

    void forward(const uint8_t *board, const int applePos, Matrix &out)
    {
        // hidden = board @ weight0
        hidden.zeros();
        for (int i = 0; i < size * size; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                hidden.values[j] += board[i] * weight0.values[i * hiddenSize + j];
            }
        }

        // hidden = activation(hidden + weight1[applePos])
        for (int j = 0; j < hiddenSize; j++)
        {
            const float x = hidden.values[j] + weight1.values[applePos * hiddenSize + j];

            if (x < -1.0f)
            {
                hidden.values[j] = -1.0f;
            }
            else if (x > 1.0f)
            {
                hidden.values[j] = 1.0f;
            }
            else
            {
                hidden.values[j] = 2.0f * x / (x * x + 1.0f);
            }
        }

        // out = hidden @ weight2
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                out.values[j] += hidden.values[i] * weight2.values[i * hiddenSize + j];
            }
        }
    }
};

#endif