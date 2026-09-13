import random, glob, os, sys
import torch
import torch.nn as nn


# ビット配列と、その解のトレーニングデータ
class TrainingData:
    def __init__(self, label, bits, filename):
        self.label, self.bits, self.filename = label, bits, filename


# 引数ファイルをロードし、１組のトレーニングデータを返す。
def load_TrainingData(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.read().splitlines()

    if len(lines) != 16 or any(len(line) != 16 for line in lines):
        raise ValueError(filename + " は16文字×16行にしてください")

    bits = [[ch == "■" for ch in line] for line in lines]
    name = os.path.basename(filename)

    # ファイル名が1_なら、条件を満たす画像
    label = name.startswith("1_")

    return TrainingData(label, bits, name)

# bitsを256個のTensorに変換
def bits_to_tensor(bits):
    values = []
    for y in range(16):
        for x in range(16):
            values.append(1.0 if bits[y][x] else 0.0)

    return torch.tensor(values, dtype=torch.float32)


# 判定対象のビット配列から予測値を求める
def predict(bits):
    input_tensor = bits_to_tensor(bits)

    prediction = linear(input_tensor)

    return prediction

# dataフォルダ下のトレーニングデータを展開
base = os.path.dirname(os.path.abspath(__file__))
TrainingDatas = [
    load_TrainingData(f)
    for f in sorted(glob.glob(os.path.join(base, "data2", "*.txt")))
]


# --------------------------------------------------
# PyTorch
# --------------------------------------------------

# 16×16 = 256個の入力値から、1個の予測値を出す
linear = nn.Linear(256, 1)

# 手作り版と同じように weight と bias を 0 から開始
with torch.no_grad():
    linear.weight.zero_()
    linear.bias.zero_()


print("===== 学習前 =====")

for s in TrainingDatas:
    target = 0.0 if s.label else 1.0

    prediction = predict(s.bits)

    print(
        s.filename,
        "教師=", s.label,
        "非合致度=", round(prediction.item(), 4)
    )


# 補正の係数
learning_rate = 0.01

# 学習の繰り返し回数
epochs = 1000


# Loss
loss_function = nn.MSELoss()


# weight と bias を SGD で更新する
optimizer = torch.optim.SGD(
    linear.parameters(),
    lr=learning_rate
)


for epoch in range(epochs):

    total_loss = 0.0

    random.shuffle(TrainingDatas)

    for s in TrainingDatas:

        # ----------------------------------------
        # ①予測
        # ----------------------------------------
        prediction = predict(s.bits)


        # 合致すれば target=0
        # 合致しなければ target=1
        target_value = 0.0 if s.label else 1.0

        target = torch.tensor(
            [target_value],
            dtype=torch.float32
        )


        # ----------------------------------------
        # 前回のgradientを0にする
        # ----------------------------------------
        optimizer.zero_grad()


        # ----------------------------------------
        # ②Loss
        # ----------------------------------------
        loss = loss_function(prediction, target)

        total_loss += loss.item()


        # ----------------------------------------
        # ③Lossを微分
        # ----------------------------------------
        loss.backward()


        # ----------------------------------------
        # ④256個のweightを補正
        # ⑤biasを補正
        # ----------------------------------------
        optimizer.step()


    if epoch == 0 or (epoch + 1) % 100 == 0:
        print(
            "epoch =", epoch + 1,
            "loss =",
            round(total_loss / len(TrainingDatas), 6)
        )


print("\n===== 学習後 =====")

for s in TrainingDatas:

    prediction = predict(s.bits)

    print(
        s.filename,
        "教師値=", s.label,
        "非合致度=", round(prediction.item(), 4)
    )


# 引数で指定されたファイルを推論
if len(sys.argv) >= 2:

    test_data = load_TrainingData(sys.argv[1])

    print("\n===== 評価画像 =====")

    for row in test_data.bits:
        print("".join("■" if v else " " for v in row))


    p = predict(test_data.bits).item()


    print("\nファイル =", sys.argv[1])
    print("非合致度 =", round(p, 4))


    # 今回は「1」が0側
    print(
        "判定 =",
        "true" if p < 0.5 else "false"
    )
