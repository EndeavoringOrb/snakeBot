#include "game.hpp"
#include "customUtils.hpp"

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

SnakeActions sampleAction(Matrix &out, uint32_t &randSeed)
{
    out.softmax();
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

float testModel(const SnakeGame &game, SnakeModel &model, Matrix &out, uint32_t &randSeed, const int iters, const int appleTolerance)
{
    // Copy of game for test runs
    SnakeGame newGame = SnakeGame(game.size, randSeed);
    float totalScore = 0.0f;

    for (int i = 0; i < iters; i++)
    {
        // Reset game state
        newGame.copyState(game);

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
                gameOver = true;
            }
            numSteps++;
        }

        // Accumulate score (apples per step)
        totalScore += (float)newGame.score / (float)numSteps;
    }

    return totalScore / (float)iters;
}

int main()
{
    // Init game stuff
    constexpr int gameSize = 16;
    float tickSpeed = 0.00f;
    sf::Clock gameClock;
    uint32_t randSeed = 42;
    SnakeGame game = SnakeGame(gameSize, randSeed);

    // Init neural network stuff
    int nTrials = 10000;
    int itersPerTrial = 100;
    float sigma = 0.01f;
    float learningRate = 0.01f;
    int appleTolerance = 300;
    const int hiddenSize = 4;

    SnakeModel model = SnakeModel(gameSize, hiddenSize);
    SnakeModel modelCopy = SnakeModel(gameSize, hiddenSize);
    Matrix weight0Grad = Matrix(gameSize * gameSize, hiddenSize);
    Matrix weight1Grad = Matrix(gameSize * gameSize, hiddenSize);
    Matrix weight2Grad = Matrix(hiddenSize, 3);
    Matrix out = Matrix(1, 3);

    // Init tracker stuff
    int stepNum = 0;
    int logInterval = 100;

    std::cout << "Model has " << model.getNumParams() << " parameters" << std::endl;

    while (true)
    {
        // Zero gradient
        weight0Grad.zeros();
        weight1Grad.zeros();
        weight2Grad.zeros();

        // Init trackers
        float totalScore = 0.0f;

        std::cout << std::endl;
        for (int i = 0; i < nTrials; i++)
        {
            if (i % logInterval == 0)
            {
                clearLines(1);
                std::cout << "Doing trial [" << i << "/" << nTrials << "]" << std::endl;
            }

            modelCopy.copyWeights(model);
            // Add random noise to copy of model using sigma
            modelCopy.addRand(randSeed, sigma);

            // Test model
            const float score = testModel(game, modelCopy, out, randSeed, itersPerTrial, appleTolerance);
            totalScore += score;

            // Update gradient
            modelCopy.weight0.mul(score);
            modelCopy.weight1.mul(score);
            modelCopy.weight2.mul(score);
            weight0Grad.add(modelCopy.weight0);
            weight1Grad.add(modelCopy.weight1);
            weight2Grad.add(modelCopy.weight2);
        }

        clearLines(1);
        std::cout << "Step " << stepNum << ", Avg. Score: " << totalScore / nTrials << std::endl;

        // Finalize gradient
        float mulVal = learningRate / (nTrials * sigma);
        weight0Grad.mul(mulVal);
        weight1Grad.mul(mulVal);
        weight2Grad.mul(mulVal);

        // Update model using gradient
        model.weight0.add(weight0Grad);
        model.weight1.add(weight1Grad);
        model.weight2.add(weight2Grad);

        stepNum++;
    }

    return 0;
}