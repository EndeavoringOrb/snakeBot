#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#include "game.hpp"

std::vector<float> getScores(const SnakeGame &game, uint32_t &randSeed, const int iters)
{
    // Counters for scores gotten
    int turnLeftScore = 0;
    int turnRightScore = 0;
    int noTurnScore = 0;

    // Copy of game for test runs
    SnakeGame newGame = SnakeGame(game.size, randSeed);

    for (int i = 0; i < iters; i++)
    {
        // Play left turn
        newGame.copyState(game);
        bool gameOver = newGame.step(SnakeActions::TURN_LEFT, randSeed);
        while (!gameOver)
        {
            gameOver = newGame.step(randAction(randSeed), randSeed);
        }
        turnLeftScore += newGame.score;

        // Play right turn
        newGame.copyState(game);
        gameOver = newGame.step(SnakeActions::TURN_RIGHT, randSeed);
        while (!gameOver)
        {
            gameOver = newGame.step(randAction(randSeed), randSeed);
        }
        turnRightScore += newGame.score;

        // Play no turn
        newGame.copyState(game);
        gameOver = newGame.step(SnakeActions::NO_TURN, randSeed);
        while (!gameOver)
        {
            gameOver = newGame.step(randAction(randSeed), randSeed);
        }
        noTurnScore += newGame.score;
    }

    std::vector<float> scores = {(float)turnLeftScore / (float)iters,
                                 (float)turnRightScore / (float)iters,
                                 (float)noTurnScore / (float)iters};
    return scores;
}

int main()
{
    // Init window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Snake Bot");
    // window.setFramerateLimit(60);

    // Init text
    sf::Font font;
    if (!font.loadFromFile("resources/fonts/arial/arial.ttf"))
    {
        // Handle font loading error
        return -1;
    }
    sf::Text infoText;
    infoText.setFont(font);
    infoText.setCharacterSize(18);
    infoText.setFillColor(sf::Color::White);
    infoText.setPosition(10, 10);

    // Init game
    constexpr int gameSize = 16;
    float tickSpeed = 0.00f;
    sf::Clock gameClock;
    uint32_t randSeed = 42;
    SnakeGame game = SnakeGame(gameSize, randSeed);

    int iters = 50000;
    int itersDelta = 500;
    std::cout << "Searching with " << iters << " iters" << std::endl;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Equal || event.key.code == sf::Keyboard::Add) {
                    iters += itersDelta;
                    std::cout << "Searching with " << iters << " iters" << std::endl;
                }
                else if (event.key.code == sf::Keyboard::Hyphen || event.key.code == sf::Keyboard::Subtract) {
                    iters -= itersDelta;
                    std::cout << "Searching with " << iters << " iters" << std::endl;
                }
            }
        }

        // Update game
        std::vector<float> scores = getScores(game, randSeed, iters);
        SnakeActions currentAction;
        if (scores[0] > scores[1] && scores[0] > scores[2])
        {
            currentAction = SnakeActions::TURN_LEFT;
        }
        else if (scores[1] > scores[0] && scores[1] > scores[2])
        {
            currentAction = SnakeActions::TURN_RIGHT;
        }
        else
        {
            currentAction = SnakeActions::NO_TURN;
        }

        bool gameOver = game.step(currentAction, randSeed);
        if (gameOver)
        {
            game.reset(randSeed);
        }

        // Update info text
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(5);
        infoText.setString(ss.str());

        // Draw
        window.clear(sf::Color::Black);

        game.render(window, infoText);

        window.draw(infoText);

        window.display();
    }

    return 0;
}