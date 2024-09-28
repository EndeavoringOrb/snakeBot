#ifndef NEURAL_NET_HPP

#include <iomanip>
#include <iostream>
#include <cmath>
#include <fstream>

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

    void mul(const float val)
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] *= val;
        }
    }

    void add(const Matrix &other)
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] += other.values[i];
        }
    }

    void sub(const Matrix &other)
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] -= other.values[i];
        }
    }

    void addRand(uint32_t &randSeed, const float std)
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] += randDist(0.0f, std, randSeed);
        }
    }

    void setRand(uint32_t &randSeed, const float std)
    {
        for (int i = 0; i < numValues; i++)
        {
            values[i] = randDist(0.0f, std, randSeed);
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
        float maxVal = values[0];

        for (int i = 0; i < numValues; i++)
        {
            maxVal = std::max(maxVal, values[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < numValues; i++)
        {
            values[i] = std::exp(values[i] - maxVal);
            sum += values[i];
        }

        // Normalize
        for (int i = 0; i < numValues; i++)
        {
            values[i] /= sum;
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

    void setRand(uint32_t &randSeed, const float std)
    {
        weight0.setRand(randSeed, std);
        weight1.setRand(randSeed, std);
        weight2.setRand(randSeed, std);
    }

    // Serialize the model to a binary file
    bool saveToFile(const std::string &filename) const
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
            return false;
        }

        // Write size and hiddenSize
        file.write(reinterpret_cast<const char *>(&size), sizeof(int));
        file.write(reinterpret_cast<const char *>(&hiddenSize), sizeof(int));

        // Write weight matrices
        file.write(reinterpret_cast<const char *>(weight0.values), weight0.numValues * sizeof(float));
        file.write(reinterpret_cast<const char *>(weight1.values), weight1.numValues * sizeof(float));
        file.write(reinterpret_cast<const char *>(weight2.values), weight2.numValues * sizeof(float));

        file.close();
        return true;
    }

    // Deserialize the model from a binary file
    static SnakeModel loadFromFile(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Error: Unable to open file for reading: " + filename);
        }

        int loadedSize, loadedHiddenSize;
        file.read(reinterpret_cast<char *>(&loadedSize), sizeof(int));
        file.read(reinterpret_cast<char *>(&loadedHiddenSize), sizeof(int));

        SnakeModel loadedModel(loadedSize, loadedHiddenSize);

        // Read weight matrices
        file.read(reinterpret_cast<char *>(loadedModel.weight0.values), loadedModel.weight0.numValues * sizeof(float));
        file.read(reinterpret_cast<char *>(loadedModel.weight1.values), loadedModel.weight1.numValues * sizeof(float));
        file.read(reinterpret_cast<char *>(loadedModel.weight2.values), loadedModel.weight2.numValues * sizeof(float));

        file.close();
        return loadedModel;
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
        //hidden.print("hidden");

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
                hidden.values[j] = (x + x) / (x * x + 1.0f);
            }
        }

        //hidden.print("hidden");

        // out = hidden @ weight2
        out.zeros();
        for (int i = 0; i < hiddenSize; i++)
        {
            out.values[0] += hidden.values[i] * weight2.values[i * hiddenSize + 0];
            out.values[1] += hidden.values[i] * weight2.values[i * hiddenSize + 1];
            out.values[2] += hidden.values[i] * weight2.values[i * hiddenSize + 2];
        }

        //out.print("out");
    }
};

#endif