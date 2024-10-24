#include "game.hpp"
#include "customUtils.hpp"
#include <filesystem>

namespace fs = std::filesystem;

int getNextTrainingRun(const std::string &directory)
{
    std::vector<int> runNumbers;

    try
    {
        // Check if directory exists
        if (!fs::exists(directory))
        {
            std::cout << "Directory doesn't exist, creating it..." << std::endl;
            fs::create_directory(directory);
            return 1; // First run
        }

        // Iterate through directory entries
        for (const auto &entry : fs::directory_iterator(directory))
        {
            try
            {
                // Convert directory name to integer
                int runNumber = std::stoi(entry.path().filename().string());
                runNumbers.push_back(runNumber);
            }
            catch (const std::invalid_argument &)
            {
                // Skip entries that can't be converted to integer
                continue;
            }
        }

        // If no valid numbered directories found
        if (runNumbers.empty())
        {
            return 0; // First run
        }

        // Find maximum number and add 1
        return *std::max_element(runNumbers.begin(), runNumbers.end()) + 1;
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        throw;
    }
}

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
    // Settings
    constexpr int gameSize = 16;

    constexpr int nTrials = 30;
    constexpr int itersPerTrial = 10000;
    float sigma = 1e-3f;
    float learningRate = 1e-2f;
    int appleTolerance = gameSize * gameSize;
    const int hiddenSize = 32;

    // Get training run ID
    int currentTrainingRun = getNextTrainingRun("trainingRuns");
    std::string currentTrainingRunPath = "trainingRuns/" + std::to_string(currentTrainingRun);

    // Create directory for training run
    if (fs::create_directories(currentTrainingRunPath))
    {
        std::cout << "Directory created successfully: " << currentTrainingRunPath << std::endl;
    }
    else
    {
        std::cerr << "Failed to create directory: " << currentTrainingRunPath << std::endl;
    }

    // Save config
    std::ofstream file(currentTrainingRunPath + "/config.txt", std::ios::out); // Open in append mode
    if (file.is_open())
    {
        file << "gameSize: " << gameSize << "\n";
        file << "nTrials: " << nTrials << "\n";
        file << "itersPerTrial: " << itersPerTrial << "\n";
        file << "sigma: " << sigma << "\n";
        file << "learningRate: " << learningRate << "\n";
        file << "appleTolerance: " << appleTolerance << "\n";
        file << "hiddenSize: " << hiddenSize << "\n";
        file.close();
    }

    std::cout << "Initializing game" << std::endl;
    // Init game stuff
    sf::Clock gameClock;
    uint32_t gameRandSeed = 42;
    uint32_t randSeed = 42;
    SnakeGame game = SnakeGame(gameSize, gameRandSeed);
    std::cout << "Initialized game" << std::endl;

    // Init neural network stuff
    std::string savePath = currentTrainingRunPath + "/model.bin";
    SnakeModel model = SnakeModel(gameSize, hiddenSize);
    SnakeModel originalModel = SnakeModel(gameSize, hiddenSize);
    originalModel.copyWeights(model);
    SnakeModel modelCopy = SnakeModel(gameSize, hiddenSize);
    Matrix weight0Grad = Matrix(gameSize * gameSize, hiddenSize);
    Matrix weight1Grad = Matrix(gameSize * gameSize, hiddenSize);
    Matrix weight2Grad = Matrix(hiddenSize, 3);
    AdamOptimizer optim0 = AdamOptimizer(weight0Grad.numValues, learningRate);
    AdamOptimizer optim1 = AdamOptimizer(weight1Grad.numValues, learningRate);
    AdamOptimizer optim2 = AdamOptimizer(weight2Grad.numValues, learningRate);
    Matrix out = Matrix(1, 3);
    std::cout << "Initialized model" << std::endl;

    float *scores = new float[nTrials];

    // Init tracker stuff
    int stepNum = 0;
    int logInterval = 1000;

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

            // Add random noise to copy of model using sigma
            modelCopy.copyWeights(model);
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
        float mulVal = 1.0f / (nTrials * sigma);
        weight0Grad.mul(mulVal);
        weight1Grad.mul(mulVal);
        weight2Grad.mul(mulVal);

        // Adam
        optim0.getGrads(weight0Grad);
        optim1.getGrads(weight1Grad);
        optim2.getGrads(weight2Grad);

        // Update model using gradient
        model.weight0.add(weight0Grad);
        model.weight1.add(weight1Grad);
        model.weight2.add(weight2Grad);

        // Print grad norm
        float norm = weight0Grad.normSquared() + weight1Grad.normSquared() + weight2Grad.normSquared();
        norm = sqrt(norm);
        std::cout << "Grad Norm: " << norm << std::endl;

        // Print distance from starting weights
        float dist = model.weight0.diffSquared(originalModel.weight0) + model.weight1.diffSquared(originalModel.weight1) + model.weight2.diffSquared(originalModel.weight2);
        dist = sqrt(dist);
        std::cout << "Current weights distance from starting weights: " << dist << std::endl;

        // Test updated model
        uint32_t testGameSeed = 42;
        const float testScore = testModel(game, model, out, testGameSeed, itersPerTrial, appleTolerance);
        std::cout << "Model Score: " << testScore << "\n\n";

        // Log
        std::ofstream logFile(currentTrainingRunPath + "/log.txt", std::ios::app); // Open in append mode
        if (logFile.is_open())
        {
            logFile << testScore << " ";
            logFile << norm << " ";
            logFile << dist << "\n";
            logFile.close();
        }

        stepNum++;

        model.saveToFile(savePath);
    }

    return 0;
}