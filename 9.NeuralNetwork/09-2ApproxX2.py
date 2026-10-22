# 09-2ApproxX2.py ― y = x**2.0 を、一次式と ReLU だけで近似する
# Python の標準ライブラリだけで動く（Colab でそのまま実行できる）
import random

# 学習データ：x を -2〜2 の41点、正解は x の2乗
xs = []
ts = []
for i in range(41):
    x = -2 + 0.1 * i
    xs.append(x)
    ts.append(x ** 2.0)


def relu(z):
    if z > 0:
        return z
    return 0.0


def predict(W1, b1, W2, b2, x):
    y = b2
    for i in range(len(W1)):
        z = W1[i] * x + b1[i]       # 隠れ層：一次式
        h = relu(z)                 #         ReLU
        y = y + W2[i] * h           # 出力層：倍率をかけて足す
    return y


def calc_mse(W1, b1, W2, b2):
    total = 0.0
    for i in range(len(xs)):
        error = predict(W1, b1, W2, b2, xs[i]) - ts[i]
        total = total + error ** 2
    return total / len(xs)


def init(H, seed):
    random.seed(seed)
    W1 = []
    b1 = []
    W2 = []
    for i in range(H):
        W1.append(random.uniform(-1, 1))
    for i in range(H):
        b1.append(random.uniform(-1, 1))
    for i in range(H):
        W2.append(random.uniform(-1, 1) / H ** 0.5)
    b2 = 0.0
    return W1, b1, W2, b2


def train(W1, b1, W2, b2, learning_rate=0.02, epochs=20000, report=()):
    H = len(W1)
    N = len(xs)

    for epoch in range(epochs + 1):

        if epoch in report:
            show(W1, b1, W2, b2, epoch)

        # 勾配を入れる場所（全データ分を足し合わせる）
        gW1 = [0.0] * H
        gb1 = [0.0] * H
        gW2 = [0.0] * H
        gb2 = 0.0

        for n in range(N):
            x = xs[n]
            target = ts[n]

            # 1. 予測
            y = b2
            z = [0.0] * H
            h = [0.0] * H
            for i in range(H):
                z[i] = W1[i] * x + b1[i]
                h[i] = relu(z[i])
                y = y + W2[i] * h[i]

            # 2. Loss（y - target）**2 を y で微分
            dy = 2 * (y - target) / N

            # 3. 各係数で微分
            gb2 = gb2 + dy
            for i in range(H):
                gW2[i] = gW2[i] + dy * h[i]
                if z[i] > 0:                              # ReLU が働いているときだけ
                    gW1[i] = gW1[i] + dy * W2[i] * x
                    gb1[i] = gb1[i] + dy * W2[i]

        # 4. 係数を補正
        for i in range(H):
            W1[i] = W1[i] - learning_rate * gW1[i]
            b1[i] = b1[i] - learning_rate * gb1[i]
            W2[i] = W2[i] - learning_rate * gW2[i]
        b2 = b2 - learning_rate * gb2

    return W1, b1, W2, b2


def show(W1, b1, W2, b2, epoch):
    print(f"  {epoch:6d}回目  mse={calc_mse(W1, b1, W2, b2):8.5f}  y(-2..2)=", end="")
    for x in (-2, -1, 0, 1, 2):
        print(f" {predict(W1, b1, W2, b2, x):5.2f}", end="")
    print()


def knots(W1, b1):
    result = []
    for i in range(len(W1)):
        if W1[i] != 0:
            result.append(-b1[i] / W1[i])      # 折れ目の位置
    result.sort()
    return result


if __name__ == "__main__":
    # 1. 手で決めた係数（4個のニューロンで5点が x**2 と一致する）
    print("【手で決めた係数】 x, 正解, 予測")
    for x in (-2, -1, 0, 1, 2):
        print(f"  {x:3d} {x ** 2:6.2f} {predict([1, 1, 1, 1], [2, 1, 0, -1], [-3, 2, 2, 2], 4, x):6.2f}")

    # 2. 学習の過程（ニューロン8個）
    print("【学習の過程】 H=8（正解 y(-2..2)=  4.00  1.00  0.00  1.00  4.00）")
    W1, b1, W2, b2 = init(8, seed=0)
    print("  学習前の折れ目 x:", " ".join(f"{k:5.2f}" for k in knots(W1, b1)))
    W1, b1, W2, b2 = train(W1, b1, W2, b2, report=(0, 100, 1000, 5000, 20000))
    print("  学習後の折れ目 x:", " ".join(f"{k:5.2f}" for k in knots(W1, b1)))

    # 3. ニューロン数と最終誤差
    print("【ニューロン数 H と最終誤差（3回試行の最小〜最大、10000回）】")
    for H in (1, 2, 4, 16):
        errors = []
        for seed in range(3):
            W1, b1, W2, b2 = init(H, seed)
            W1, b1, W2, b2 = train(W1, b1, W2, b2, epochs=10000)
            errors.append(calc_mse(W1, b1, W2, b2))
        print(f"  H={H:2d}  {min(errors):.5f} 〜 {max(errors):.5f}")
