#include "game.hpp"
#include "customUtils.hpp"

int main()
{
    // Init window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Snake Bot");
    window.setFramerateLimit(60);

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

    // Load model
    SnakeModel model = SnakeModel(1, 1).loadFromFile("model.bin");
    Matrix out = Matrix(1, 3);

    // Init game
    float tickSpeed = 0.2f;
    sf::Clock gameClock;
    uint32_t randSeed = 42;
    SnakeGame game = SnakeGame(model.size, randSeed);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        if (gameClock.getElapsedTime().asSeconds() > tickSpeed)
        {
            // Model forward
            model.forward(game.board, game.applePosition, out);

            // Update game
            bool gameOver = game.step(sampleAction(out, randSeed), randSeed);
            if (gameOver)
            {
                game.reset(randSeed);
            }

            gameClock.restart();
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