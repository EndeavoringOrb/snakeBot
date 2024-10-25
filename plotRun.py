import matplotlib.pyplot as plt
import numpy as np


def plotRun(num):
    try:
        num = int(num)

        folder = f"trainingRuns/{num}"

        with open(f"{folder}/config.txt", "r", encoding="utf-8") as f:
            configLines = f.read().strip().split("\n")
            config = {}
            for line in configLines:
                line = line.split(": ")
                try:
                    config[line[0]] = float(line[1])
                except Exception as e:
                    config[line[0]] = line[1]

            for k, v in config.items():
                print(f"  {k}: {v}")

        with open(f"{folder}/log.txt", "r", encoding="utf-8") as f:
            lines = [item.strip().split(" ") for item in f.read().strip().split("\n")]

        values = [float(item[0]) for item in lines]
        ticks = np.arange(len(values)) * config["nTrials"] * config["itersPerTrial"]

        plt.plot(ticks, values, label=f"Run {num}")

        return True

    except Exception as e:
        return False


validRun = True
while validRun:
    validRun = plotRun(input("Enter training run #: "))

plt.ylabel("Avg. Reward (# Apples Eaten)")
plt.xlabel("# Games Played")
plt.legend()
plt.show()
