import matplotlib.pyplot as plt


def plotRun(num):
    try:
        num = int(num)
    except Exception as e:
        return False

    folder = f"trainingRuns/{num}"

    with open(f"{folder}/config.txt", "r", encoding="utf-8") as f:
        config = f.read().strip()
        for line in config.split("\n"):
            print(f"  {line}")

    with open(f"{folder}/log.txt", "r", encoding="utf-8") as f:
        lines = [item.strip().split(" ") for item in f.read().strip().split("\n")]

    values = [float(item[0]) for item in lines]

    plt.plot(values, label=f"Run {num}")

    return True


validRun = True
while validRun:
    validRun = plotRun(input("Enter training run #: "))

plt.ylabel("Avg. Reward (# Apples Eaten)")
plt.xlabel("Training Step")
plt.legend()
plt.show()
