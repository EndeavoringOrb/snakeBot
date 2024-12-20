#ifndef GAME_HPP

#include <iostream>

#include "neuralNet.hpp"
#include "customUtils.hpp"

enum SnakeDirections
{
    LEFT,
    UP,
    RIGHT,
    DOWN
};

enum SnakeActions
{
    TURN_LEFT,
    TURN_RIGHT,
    NO_TURN
};

SnakeActions randAction(uint32_t &randSeed)
{
    const float randVal = randFloat(randSeed);
    return (SnakeActions)std::min(2, (int)(3.0f * randVal));
}

SnakeActions sampleAction(Matrix &out, uint32_t &randSeed)
{
    out.softmax();
    // out.print("probs");
    const float val = randFloat(randSeed);
    if (val < out.values[0])
    {
        return SnakeActions::TURN_LEFT;
    }
    if (val < out.values[0] + out.values[1])
    {
        return SnakeActions::TURN_RIGHT;
    }
    return SnakeActions::NO_TURN;
};

struct SnakeGame
{
    uint8_t *board;
    int applePosition;

    int size;
    int snakeHeadPosition;
    int snakeDirection = SnakeDirections::RIGHT;
    int score = 0;

    SnakeGame(int _size, uint32_t &randSeed)
    {
        size = _size;
        board = new uint8_t[size * size];
        reset(randSeed);
    }

    ~SnakeGame()
    {
        delete[] board;
    }

    void reset(uint32_t &randSeed)
    {
        // Reset board
        for (int i = 0; i < size * size; i++)
        {
            board[i] = false;
        }

        // Reset snake
        snakeHeadPosition = (size / 2) * size + size / 2;
        board[snakeHeadPosition] = 2;
        board[snakeHeadPosition - 1] = 1;
        snakeDirection = SnakeDirections::RIGHT;

        // Reset score
        score = 0;

        // Reset apple position
        randomizeApplePosition(randSeed);
    }

    void randomizeApplePosition(uint32_t &randSeed)
    {
        applePosition = randInt(randSeed, size * size);
        while (board[applePosition] > 0)
        {
            applePosition = randInt(randSeed, size * size);
        }
    }

    bool step(SnakeActions action, uint32_t &randSeed)
    {
        // Update direction
        if (action == SnakeActions::TURN_LEFT)
        {
            snakeDirection = (snakeDirection + 4 - 1) % 4;
        }
        else if (action == SnakeActions::TURN_RIGHT)
        {
            snakeDirection = (snakeDirection + 4 + 1) % 4;
        }

        // Move snake
        int newHeadPosition;
        bool hitSomething = false;
        if (snakeDirection == SnakeDirections::LEFT)
        {
            newHeadPosition = snakeHeadPosition - 1;
            hitSomething = (newHeadPosition % size == size - 1);
        }
        else if (snakeDirection == SnakeDirections::UP)
        {
            newHeadPosition = snakeHeadPosition - size;
            hitSomething = (newHeadPosition < 0);
        }
        else if (snakeDirection == SnakeDirections::RIGHT)
        {
            newHeadPosition = snakeHeadPosition + 1;
            hitSomething = (newHeadPosition % size == 0);
        }
        else if (snakeDirection == SnakeDirections::DOWN)
        {
            newHeadPosition = snakeHeadPosition + size;
            hitSomething = (newHeadPosition >= size * size);
        }

        if (hitSomething)
        {
            return true;
        }

        // Check for snake collisions
        if (board[newHeadPosition] > 1) // Greater than 1 because we have not updated the tail yet, so the tail (value == 1) will be moved when we move the head to this position
        {
            return true;
        }

        // Update head position
        snakeHeadPosition = newHeadPosition;

        // Apple collision
        if (snakeHeadPosition == applePosition)
        {
            // Increase score
            score++;

            // Got max score
            if (score + 2 == size * size)
            {
                return true;
            }

            // Update board with new head pos
            board[snakeHeadPosition] = score + 2; // +2 because we start with a length of 2
            randomizeApplePosition(randSeed);
        }
        else
        {
            // Move snake, Remove back of snake
            uint8_t *boardptr = board;
            uint8_t *end = boardptr + (size * size - 1);
            for (; boardptr <= end; boardptr++)
            {
                if (*boardptr > 0)
                {
                    (*boardptr)--;
                }
            }

            // Update board with new head pos
            board[snakeHeadPosition] = score + 2; // +2 because we start with a length of 2
        }

        return false;
    }

    void copyState(const SnakeGame &other)
    {
        for (int i = 0; i < size * size; i++)
        {
            board[i] = other.board[i];
        }
        applePosition = other.applePosition;
        snakeHeadPosition = other.snakeHeadPosition;
        snakeDirection = other.snakeDirection;
        score = other.score;
    }

    void print()
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                const int index = i * size + j;
                if (index == applePosition)
                {
                    std::cout << "A" << " ";
                }
                else
                {
                    std::cout << std::to_string(board[index]) << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    void render()
    {
        clearLines(size + 1);
        std::cout << "Score: " << score << std::endl;
        // Draw the board
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                const int index = i * size + j;
                if (index == applePosition)
                {
                    // Draw apple
                    std::cout << "@";
                }
                else if (board[index] > 0)
                {
                    // Draw snake
                    std::cout << "#";
                }
                else
                {
                    // Draw empty cell
                    std::cout << ".";
                }
            }
            std::cout << std::endl;
        }
    }
};

#endif