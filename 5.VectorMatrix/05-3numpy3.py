import os
import random
import numpy as np


# ------------------------------------
# Training Data
# ------------------------------------

class TrainingData:
    def __init__(self, dir, filename):
        self.filename = filename

        # 0 = |
        # 1 = -
        # 2 = /
        self.class_id = int(filename[0])

        with open(dir + "/" + filename, encoding="utf-8") as f:
            lines = f.read().splitlines()

        if len(lines) != 16:
            raise ValueError(filename + " は16行にしてください")

        self.bits = []

        for line in lines:
            if len(line) != 16:
                raise ValueError(filename + " は各行16文字にしてください")

            for ch in line:
                self.bits.append(1.0 if ch == "■" else 0.0)


# ------------------------------------
# Training Data 読み込み
# ------------------------------------
training_data = []
data_dir = r"data3"
files = os.listdir(data_dir)

for filename in files:
    if len(filename) > 0 and filename[0] in "012":
        training_data.append(
            TrainingData(data_dir, filename)
        )


# ------------------------------------
# X 15ファイル分のビット構成を作る
# 15件 × 256入力
# ------------------------------------
X = np.array([
    td.bits for td in training_data
])

print("X.shape =", X.shape)


# ------------------------------------
# target を作る　各ファイルの正解位置に1.0をセット
# 15件 × 3クラス
# ------------------------------------
targets = np.zeros((len(training_data), 3))

for i, td in enumerate(training_data):
    targets[i][td.class_id] = 1.0

print("targets.shape =", targets.shape)

# ------------------------------------
# weight と biasの初期値設定
# ------------------------------------
weights = np.zeros((3, 256))
bias = np.zeros(3)

print("weights.shape =", weights.shape)
print("bias.shape =", bias.shape)

# ------------------------------------
# 学習
# ------------------------------------
rate = 0.04 / 4
epochs = 600


for epoch in range(epochs):

    # ----------------------------
    # Prediction　(15×256) @ (256×3)　> (15×3)
    # ----------------------------
    prediction = X @ weights.T + bias

    # ----------------------------
    # diff 15x3 - 15x3 = 15x
    # ----------------------------
    diff = prediction - targets

    # ----------------------------
    # Loss 15x3 = float(単一)
    # ----------------------------
    loss = np.sum(diff ** 2)

    # ----------------------------
    # Prediction に対する gradient
    #　15x3 * 単一　> 15x3
    # d/dPrediction (diff^2)
    # = 2 * diff　
    # ----------------------------
    gradient = 2 * diff

    # ----------------------------
    # weight の gradient
    # 15x3.T=(3x15) @ (3x256) > 3x256
    # ----------------------------
    weight_gradient = gradient.T @ X

    # ----------------------------
    # bias の gradient
    # 15件分を合計
    # ----------------------------
    bias_gradient = np.sum(
        gradient,
        axis=0
    )

    # ----------------------------
    # weight / bias 更新
    # ----------------------------
    weights -= rate * weight_gradient
    bias -= rate *  bias_gradient

    # ----------------------------
    # Loss表示
    # ----------------------------
    if epoch % 50 == 0:
        print(
            "epoch =",
            epoch,
            "loss =",
            loss
        )


# ------------------------------------
# 学習結果
# ------------------------------------
print()
print("----- result -----")

prediction = X @ weights.T + bias
for i, td in enumerate(training_data):
    answer = np.argmax(
        prediction[i]
    )

    print(
        td.filename,
        "target =",
        td.class_id,
        "prediction =",
        prediction[i],
            "answer =",
        answer
    )