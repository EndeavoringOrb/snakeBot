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

    Matrix() {}

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

    void addOther(const Matrix &other, int start, int stop)
    {
        for (int i = start; i < stop; i++)
        {
            values[i] += other.values[i - start];
        }
    }

    void otherAdd(const Matrix &other, int start, int stop)
    {
        for (int i = start; i < stop; i++)
        {
            values[i - start] += other.values[i];
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

    float normSquared()
    {
        float val = 0.0f;
        for (int i = 0; i < numValues; i++)
        {
            val += values[i] * values[i];
        }
        return val;
    }

    float diffSquared(Matrix &other)
    {
        float val = 0.0f;
        for (int i = 0; i < numValues; i++)
        {
            float diff = values[i] - other.values[i];
            val += diff * diff;
        }
        return val;
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
        // hidden.print("hidden");

        // hidden = activation(hidden * weight1[applePos])
        for (int j = 0; j < hiddenSize; j++)
        {
            const float x = hidden.values[j] * weight1.values[applePos * hiddenSize + j];

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

        // hidden.print("hidden");

        // out = hidden @ weight2
        out.zeros();
        for (int i = 0; i < hiddenSize; i++)
        {
            out.values[0] += hidden.values[i] * weight2.values[i * 3 + 0];
            out.values[1] += hidden.values[i] * weight2.values[i * 3 + 1];
            out.values[2] += hidden.values[i] * weight2.values[i * 3 + 2];
        }

        // out.print("out");
    }
};

struct AdamOptimizer
{
    int nParams;
    Matrix m;
    Matrix v;
    float alpha;      // the learning rate. good default value: 1e-2
    float beta1;      // good default value: 0.9
    float beta1Power; // used instead of calculating a std::pow every iteration
    float beta2;      // good default value: 0.999
    float beta2Power; // used instead of calculating a std::pow every iteration
    int t = 0;
    float eps;

    AdamOptimizer(int _nParams, float _alpha, float _beta1 = 0.9f, float _beta2 = 0.999f, float _eps = 1e-5f)
        : m(1, _nParams),
          v(1, _nParams)
    {
        nParams = _nParams;
        alpha = _alpha;
        beta1 = _beta1;
        beta2 = _beta2;
        beta1Power = beta1;
        beta2Power = beta2;
        eps = _eps;
    }

    void getGrads(Matrix &grad)
    {
        // Compute constants
        const float beta1Minus = 1.0f - beta1;
        const float beta2Minus = 1.0f - beta2;
        const float mHatMul = 1.0f / (1.0f - beta1Power);
        const float vHatMul = 1.0f / (1.0f - beta2Power);
        for (int i = 0; i < nParams; i++)
        {
            // Compute m and mHat
            const float mVal = beta1 * m.values[i] + beta1Minus * grad.values[i];
            m.values[i] = mVal;
            const float mHatVal = mVal * mHatMul;

            // Compute v and vHat
            const float vVal = beta2 * v.values[i] + beta2Minus * grad.values[i] * grad.values[i];
            v.values[i] = vVal;
            const float vHatVal = vVal * vHatMul;

            // Compute new grad
            grad.values[i] = alpha * mHatVal / (sqrtf(vHatVal) + eps);
        }

        // Increase values
        beta1Power *= beta1;
        beta2Power *= beta2;
        t++;
    }
};

#endif