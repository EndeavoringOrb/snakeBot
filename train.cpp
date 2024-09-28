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
    // Init game stuff
    constexpr int gameSize = 4;
    sf::Clock gameClock;
    uint32_t gameRandSeed = 42;
    uint32_t randSeed = 42;
    SnakeGame game = SnakeGame(gameSize, gameRandSeed);

    // Init neural network stuff
    constexpr int nTrials = 1000;
    constexpr int itersPerTrial = 10000;
    float sigma = 1e-2f;
    float learningRate = 1e-3f;
    int appleTolerance = gameSize * gameSize;
    const int hiddenSize = 16;

    std::string savePath = "model.bin";
    SnakeModel model = SnakeModel(gameSize, hiddenSize);
    SnakeModel modelCopy = SnakeModel(gameSize, hiddenSize);
    Matrix weight0Grad = Matrix(gameSize * gameSize, hiddenSize);
    Matrix weight1Grad = Matrix(gameSize * gameSize, hiddenSize);
    Matrix weight2Grad = Matrix(hiddenSize, 3);
    Matrix out = Matrix(1, 3);

    float *scores = new float[nTrials];

    // Init tracker stuff
    int stepNum = 0;
    int logInterval = 10;

    std::cout << "Model has " << model.getNumParams() << " parameters" << std::endl;

    while (true)
    {
        // Zero gradient
        weight0Grad.zeros();
        weight1Grad.zeros();
        weight2Grad.zeros();

        float meanScore = 0.0f;
        uint32_t noiseSeed = randSeed;

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
            const float score = testModel(game, modelCopy, out, gameRandSeed, itersPerTrial, appleTolerance);
            scores[i] = score;
            meanScore += score;
        }

        // Get mean and std
        meanScore /= (float)nTrials;
        float std = 0.0f;
        for (int i = 0; i < nTrials; i++)
        {
            const float x = scores[i] - meanScore;
            std += x * x;
        }
        std = sqrt(std / (float)nTrials);
        const float invStd = 1.0f / std;

        clearLines(1);
        std::cout << "Step " << stepNum << ", Avg. Score: " << meanScore << std::endl;

        // Normalize scores and update gradient
        for (int i = 0; i < nTrials; i++)
        {
            const float scoreVal = (scores[i] - meanScore) * invStd;
            modelCopy.setRand(noiseSeed, sigma); // Get just the noise, not weights + noise
            modelCopy.weight0.mul(scoreVal);
            modelCopy.weight1.mul(scoreVal);
            modelCopy.weight2.mul(scoreVal);
            weight0Grad.add(modelCopy.weight0);
            weight1Grad.add(modelCopy.weight1);
            weight2Grad.add(modelCopy.weight2);
        }

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

        model.saveToFile(savePath);
    }

    return 0;
}