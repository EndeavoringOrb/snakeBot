#include "game.hpp"
#include "customUtils.hpp"

float testModel(const SnakeGame &game, SnakeModel &model, Matrix &out, uint32_t &randSeed, const int iters, const int appleTolerance)
{
    // Copy of game for test runs
    SnakeGame newGame = SnakeGame(game.size, randSeed);
    float totalScore = 0.0f;
    float maxScore = 0.0f;

    for (int i = 0; i < iters; i++)
    {
        // Reset game state
        newGame.copyState(game);
        newGame.randomizeApplePosition(randSeed);

        // Play game to end
        int numSteps = 0;
        int lastAppleStep = 0;
        bool gameOver = false;
        while (!gameOver)
        {
            // Model forward
            model.forward(newGame.board, newGame.applePosition, out);

            // Take step
            const int preStepScore = newGame.score;
            gameOver = newGame.step(sampleAction(out, randSeed), randSeed);
            if (newGame.score > preStepScore)
            {
                lastAppleStep = numSteps;
            }
            else if (numSteps - lastAppleStep > appleTolerance)
            {
                gameOver = true; // Have gone appleTolerance steps without getting an apple, so stop
            }
            numSteps++;
        }

        // Accumulate score (apples per step)
        // totalScore += (float)newGame.score / (float)numSteps;
        // maxScore = std::max(maxScore, (float)newGame.score);
        totalScore += newGame.score;
    }

    return totalScore / (float)iters;
}

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
    int trainingRun;
    std::cout << "Enter training run #: ";
    std::cin >> trainingRun;
    SnakeModel model = SnakeModel(1, 1).loadFromFile("trainingRuns/" + std::to_string(trainingRun) + "/model.bin");
    Matrix out = Matrix(1, 3);
    std::cout << "Loaded model with " << model.getNumParams() << " parameters" << std::endl;

    // Init game
    float tickSpeed = 0.2f;
    sf::Clock gameClock;
    uint32_t randSeed = 42;
    SnakeGame game = SnakeGame(model.size, randSeed);

    float score = testModel(game, model, out, randSeed, 1000, game.size * game.size);
    std::cout << "Model Avg. Score: " << score << std::endl;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::Resized)
            {
                sf::View view = window.getView();
                view.setSize(event.size.width, event.size.height);
                view.setCenter((float)event.size.width / 2.0f, (float)event.size.height / 2.0f);
                window.setView(view);
            }
        }

        if (gameClock.getElapsedTime().asSeconds() > tickSpeed)
        {
            // Model forward
            model.forward(game.board, game.applePosition, out);

            // Update game
            /*bool gameOver;
            if (out.values[0] > out.values[1] && out.values[0] > out.values[2])
            {
                gameOver = game.step(SnakeActions::TURN_LEFT, randSeed);
            }
            else if (out.values[1] > out.values[0] && out.values[1] > out.values[2])
            {
                gameOver = game.step(SnakeActions::TURN_RIGHT, randSeed);
            }
            else
            {
                gameOver = game.step(SnakeActions::NO_TURN, randSeed);
            }*/
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