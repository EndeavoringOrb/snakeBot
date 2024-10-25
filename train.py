import numpy as np

gameSize = 4
hiddenSize = 32


class SnakeModel:
    def __init__(self) -> None:
        pass

    def forward(self, weights, board, applePos):
        hidden = board @ weights[: gameSize * gameSize * hiddenSize].reshape(
            gameSize * gameSize, hiddenSize
        )
        hidden = np.tanh(
            hidden
            + weights[
                gameSize * gameSize * hiddenSize
                + applePos * hiddenSize : gameSize * gameSize * hiddenSize
                + (applePos + 1) * hiddenSize
            ]
        )
        out = hidden @ weights[2 * gameSize * gameSize * hiddenSize :]
        return out

    def getWeights(self, gameSize, hiddenSize):
        weights = np.zeros(
            gameSize * gameSize * hiddenSize
            + gameSize * gameSize * hiddenSize
            + hiddenSize * 3
        )
        return weights


model = SnakeModel()
weights = model.getWeights(gameSize, hiddenSize)

