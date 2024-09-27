#ifndef GAME_HPP

#include <SFML/Graphics.hpp>

#include <iostream>

#include "neuralNet.hpp"

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
        snakeHeadPosition = 135;
        board[135] = 3;
        board[134] = 2;
        board[133] = 1;
        snakeDirection = SnakeDirections::RIGHT;

        // Reset score
        score = 0;

        // Reset apple position
        randomizeApplePosition(randSeed);
    }

    void randomizeApplePosition(uint32_t &randSeed)
    {
        applePosition = rand256(randSeed);
        while (board[applePosition] > 0)
        {
            applePosition = rand256(randSeed);
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
            // Increase score, and increase length of snake
            score++;
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
        }

        // Add new head position
        board[snakeHeadPosition] = score + 3; // +3 because we start with a length of 3

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

    void render(sf::RenderWindow &window, sf::Text &renderText)
    {
        const float cellSize = std::min(window.getSize().x / size, window.getSize().y / size);
        const float offsetX = (window.getSize().x - cellSize * size) / 2;
        const float offsetY = (window.getSize().y - cellSize * size) / 2;

        // Clear the window
        window.clear(sf::Color::Black);

        // Draw the board
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                sf::RectangleShape cell(sf::Vector2f(cellSize - 1, cellSize - 1));
                cell.setPosition(offsetX + j * cellSize, offsetY + i * cellSize);

                const int index = i * size + j;
                if (index == applePosition)
                {
                    // Draw apple
                    cell.setFillColor(sf::Color::Red);
                }
                else if (board[index] > 0)
                {
                    // Draw snake
                    if (index == snakeHeadPosition)
                    {
                        cell.setFillColor(sf::Color::Green);
                    }
                    else
                    {
                        cell.setFillColor(sf::Color(0, 125, 0)); // Darker green for the body
                    }
                }
                else
                {
                    // Draw empty cell
                    cell.setFillColor(sf::Color(50, 50, 50)); // Dark gray
                }

                window.draw(cell);
            }
        }

        // Draw the score
        renderText.setString("Score: " + std::to_string(score));
        window.draw(renderText);
    }
};

#endif