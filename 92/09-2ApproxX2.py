# 09-2ApproxX2.py ― y = x**2.0 を、一次式と ReLU だけで近似する
# Python の標準ライブラリだけで動く（Colab でそのまま実行できる）
import random

xs = [-2 + 0.1 * i for i in range(41)]       # 入力（-2〜2 の41点）
ts = [x ** 2.0 for x in xs]                  # 正解


def relu(z):
    return max(0.0, z)


def init(H, seed):
    random.seed(seed)
    W1 = [random.uniform(-1, 1) for _ in range(H)]                # 隠れ層の傾き
    b1 = [random.uniform(-1, 1) for _ in range(H)]                # 隠れ層のずらし
    W2 = [random.uniform(-1, 1) / H ** 0.5 for _ in range(H)]     # 出力層の倍率
    b2 = 0.0                                                      # 出力層のずらし
    return W1, b1, W2, b2


def predict(params, x):
    W1, b1, W2, b2 = params
    h = [relu(w * x + b) for w, b in zip(W1, b1)]      # 隠れ層：一次式 → ReLU
    return sum(v * hi for v, hi in zip(W2, h)) + b2    # 出力層：一次式だけ


def mse(params):
    return sum((predict(params, x) - t) ** 2 for x, t in zip(xs, ts)) / len(xs)


def train(params, lr=0.02, epochs=20000, report=()):
    W1, b1, W2, b2 = params
    H, N = len(W1), len(xs)
    for epoch in range(epochs + 1):
        if epoch in report:
            show((W1, b1, W2, b2), epoch)
        gW1, gb1, gW2, gb2 = [0.0] * H, [0.0] * H, [0.0] * H, 0.0
        for x, t in zip(xs, ts):
            z = [w * x + b for w, b in zip(W1, b1)]
            h = [relu(zi) for zi in z]
            y = sum(v * hi for v, hi in zip(W2, h)) + b2
            dy = 2 * (y - t) / N                       # 誤差が y にどう効くか
            gb2 += dy
            for i in range(H):
                gW2[i] += dy * h[i]
                if z[i] > 0:                           # ReLU が働いているときだけ
                    gW1[i] += dy * W2[i] * x
                    gb1[i] += dy * W2[i]
        W1 = [w - lr * g for w, g in zip(W1, gW1)]     # 勾配と逆の向きに動かす
        b1 = [b - lr * g for b, g in zip(b1, gb1)]
        W2 = [v - lr * g for v, g in zip(W2, gW2)]
        b2 -= lr * gb2
    return W1, b1, W2, b2


def show(params, epoch):
    ys = " ".join(f"{predict(params, x):5.2f}" for x in (-2, -1, 0, 1, 2))
    print(f"  {epoch:6d}回目  mse={mse(params):8.5f}  y(-2..2)= {ys}")


def knots(params):
    W1, b1, _, _ = params
    return sorted(-b / w for w, b in zip(W1, b1) if w != 0)


if __name__ == "__main__":
    # 1. 手で決めた係数（4個のニューロンで5点が x**2 と一致する）
    hand = ([1, 1, 1, 1], [2, 1, 0, -1], [-3, 2, 2, 2], 4)
    print("【手で決めた係数】 x, 正解, 予測")
    for x in (-2, -1, 0, 1, 2):
        print(f"  {x:3d} {x ** 2:6.2f} {predict(hand, x):6.2f}")

    # 2. 学習の過程（ニューロン8個）
    print("【学習の過程】 H=8（正解 y(-2..2)= 4.00 1.00 0.00 1.00 4.00）")
    p = init(8, seed=0)
    print("  学習前の折れ目 x:", " ".join(f"{k:5.2f}" for k in knots(p)))
    p = train(p, report=(0, 100, 1000, 5000, 20000))
    print("  学習後の折れ目 x:", " ".join(f"{k:5.2f}" for k in knots(p)))

    # 3. ニューロン数と最終誤差
    print("【ニューロン数 H と最終誤差（3回試行の最小〜最大、10000回）】")
    for H in (1, 2, 4, 16):
        r = [mse(train(init(H, seed), epochs=10000)) for seed in range(3)]
        print(f"  H={H:2d}  {min(r):.5f} 〜 {max(r):.5f}")
