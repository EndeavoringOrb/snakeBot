#ifndef RANDOM_HPP

#include <cstdint>
#include <limits>

/**
 * Manipulates the input uint32_t so that the output is "random". It is not actually random: the same input will result in the same output, but given a certain input it is hard for an onlooker to predict the output without knowing the exact function.
 *
 * @param input A 32-bit unsigned integer
 * @return A new 32-bit unsigned integer
 */
static uint32_t PCG_Hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/**
 * Generates a random floating-point value in the range [0, 1]
 *
 * @param seed A reference to a 32-bit unsigned integer used as the seed for the PCG Hash. This is a reference so that we can change the seed within the function which means two subsequent calls to randFloat will return different values because the seed is changed without us having to manually update the seed.
 * @return A random floating-point value in the range [0, 1].
 */
float randFloat(uint32_t &seed)
{
    seed = PCG_Hash(seed);
    return (float)seed / (float)std::numeric_limits<uint32_t>::max();
}

/**
 * Generates a random uint8_t.
 *
 * @param seed A reference to a 32-bit unsigned integer used as the seed for the PCG Hash. This is a reference so that we can change the seed within the function which means two subsequent calls to randFloat will return different values because the seed is changed without us having to manually update the seed.
 * @return A random value in the range [0, 255].
 */
uint8_t rand256(uint32_t &seed)
{
    seed = PCG_Hash(seed);
    return seed & 0xFF;
}

/**
 * Generates a random floating-point value following a normal distribution
 *
 * @param mean A float, the mean of the normal distribution
 * @param std A float, the standard deviation of the normal distribution
 * @param seed A reference to a 32-bit unsigned integer used as the seed
 * @return A random floating-point value
 */
float randDist(const float mean, const float std, uint32_t &randSeed)
{
    // Generate two independent random numbers from a uniform distribution in the range (0,1)
    float u1 = randFloat(randSeed);
    float u2 = randFloat(randSeed);

    // Box-Muller transform to convert uniform random numbers to normal distribution
    float z0 = std * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2) + mean;

    // Return the generated random number
    return z0;
}

#endif