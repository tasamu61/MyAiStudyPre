import math, random, glob, os

class Sample:
    def __init__(self, label, bits, filename):
        self.label, self.bits, self.filename = label, bits, filename

def load_sample(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) != 16 or any(len(line) != 16 for line in lines):
        raise ValueError(filename + " は16文字×16行にしてください")
    bits = [[ch == "■" for ch in line] for line in lines]
    name = os.path.basename(filename)
    label = name.startswith("1_")
    return Sample(label, bits, name)

base = "." # os.path.dirname(os.path.abspath(__file__))
samples = [load_sample(f) for f in sorted(glob.glob(os.path.join(base, "data2", "*.txt")))]

weights = [[0.0]*16 for _ in range(16)]
bias = 0.0

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def predict(bits):
    z = bias
    for y in range(16):
        for x in range(16):
            z += (1.0 if bits[y][x] else 0.0) * weights[y][x]
    return sigmoid(z)

print("===== 学習前 =====")
for s in samples:
    print(s.filename, "正解=", int(s.label), "予測=", round(predict(s.bits), 4))

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    total_loss = 0.0
    random.shuffle(samples)
    for s in samples:
        prediction = predict(s.bits)                 # ①予測
        target = 1.0 if s.label else 0.0
        loss = (prediction - target) ** 2            # ②Loss
        total_loss += loss

        # ③微分: dLoss/dz = 2(p-t) * p(1-p)
        gradient_z = 2*(prediction-target)*prediction*(1-prediction)

        # ④256個のweightを補正
        for y in range(16):
            for x in range(16):
                input_value = 1.0 if s.bits[y][x] else 0.0
                weights[y][x] -= learning_rate * gradient_z * input_value

        # ⑤biasを補正
        bias -= learning_rate * gradient_z

    if epoch == 0 or (epoch+1) % 100 == 0:
        print("epoch =", epoch+1, "loss =", round(total_loss/len(samples), 6))

print("\n===== 学習後 =====")
for s in samples:
    print(s.filename, "正解=", int(s.label), "予測=", round(predict(s.bits), 4))

# 未知の「1」で推論
test = [[False]*16 for _ in range(16)]
test[1][7] = True
test[2][6] = test[2][7] = True
for y in range(3,14):
    test[y][7] = True
test[14][6] = test[14][7] = test[14][8] = True

print("\n===== 未知画像 =====")
for row in test:
    print("".join("■" if v else " " for v in row))

p = predict(test)
print("\n1である予測値 =", round(p,4))
print("判定 =", "1" if p >= 0.5 else "1ではない")
