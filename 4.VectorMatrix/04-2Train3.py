import os


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
# 15件 × 256入力（numpy3.pyのXと同じ中身を、リストのリストで持つ）
# ------------------------------------
X = [td.bits for td in training_data]

print("X件数 =", len(X), " 1件あたりの入力数 =", len(X[0]))


# ------------------------------------
# target を作る　各ファイルの正解位置に1.0をセット
# 15件 × 3クラス
# ------------------------------------
targets = [[0.0, 0.0, 0.0] for _ in training_data]

for i, td in enumerate(training_data):
    targets[i][td.class_id] = 1.0

print("targets件数 =", len(targets), " 1件あたりのクラス数 =", len(targets[0]))


# ------------------------------------
# weight と biasの初期値設定
# weights: 3クラス × 256入力、bias: 3クラス分
# ------------------------------------
weights = [[0.0] * 256 for _ in range(3)]
bias = [0.0, 0.0, 0.0]

print("weights: 3 x 256, bias: 3")


# ------------------------------------
# 学習
# ------------------------------------
rate = 0.04 / 4
epochs = 600


for epoch in range(epochs):

    # ----------------------------
    # Prediction　15件 × 3クラス
    # numpy3.pyの X @ weights.T + bias にあたる部分を
    # ひとつずつ掛けて足す二重ループで書く
    # ----------------------------
    prediction = [[0.0, 0.0, 0.0] for _ in range(len(X))]
    for i in range(len(X)):
        for c in range(3):
            z = bias[c]
            for j in range(256):
                z += X[i][j] * weights[c][j]
            prediction[i][c] = z

    # ----------------------------
    # diff 15件 × 3クラス
    # ----------------------------
    diff = [[prediction[i][c] - targets[i][c] for c in range(3)] for i in range(len(X))]

    # ----------------------------
    # Loss  全件・全クラス分の2乗誤差を合計した、ただ1つの数値
    # ----------------------------
    loss = 0.0
    for i in range(len(X)):
        for c in range(3):
            loss += diff[i][c] ** 2

    # ----------------------------
    # Prediction に対する gradient
    # 15件 × 3クラス
    # d/dPrediction (diff^2) = 2 * diff
    # ----------------------------
    gradient = [[2 * diff[i][c] for c in range(3)] for i in range(len(X))]

    # ----------------------------
    # weight の gradient
    # 3クラス × 256入力
    # numpy3.pyの gradient.T @ X にあたる部分
    # ----------------------------
    weight_gradient = [[0.0] * 256 for _ in range(3)]
    for c in range(3):
        for j in range(256):
            total = 0.0
            for i in range(len(X)):
                total += gradient[i][c] * X[i][j]
            weight_gradient[c][j] = total

    # ----------------------------
    # bias の gradient
    # 3クラス分、15件分を合計
    # ----------------------------
    bias_gradient = [0.0, 0.0, 0.0]
    for c in range(3):
        total = 0.0
        for i in range(len(X)):
            total += gradient[i][c]
        bias_gradient[c] = total

    # ----------------------------
    # weight / bias 更新
    # ----------------------------
    for c in range(3):
        for j in range(256):
            weights[c][j] -= rate * weight_gradient[c][j]
        bias[c] -= rate * bias_gradient[c]

    # ----------------------------
    # Loss表示
    # ----------------------------
    if epoch % 50 == 0:
        print("epoch =", epoch, "loss =", loss)


# ------------------------------------
# 学習結果
# ------------------------------------
print()
print("----- result -----")

for i, td in enumerate(training_data):
    z = [0.0, 0.0, 0.0]
    for c in range(3):
        s = bias[c]
        for j in range(256):
            s += X[i][j] * weights[c][j]
        z[c] = s

    answer = 0
    for c in range(1, 3):
        if z[c] > z[answer]:
            answer = c

    print(
        td.filename,
        "target =",
        td.class_id,
        "prediction =",
        [round(v, 3) for v in z],
        "answer =",
        answer,
    )
