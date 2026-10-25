# 09-3DeepX2.py ― 層の数・幅・関数を自由に変えて y = x**2 を近似する（NumPy版、Colabで実行可）
import numpy as np

# 活性化関数と、その微分（学習で使う）。f と d は必ずセットで替える
ACTS = {
    "relu":    (lambda z: np.maximum(0, z),         lambda z: (z > 0).astype(float)),
    "leaky":   (lambda z: np.where(z > 0, z, 0.1*z), lambda z: np.where(z > 0, 1.0, 0.1)),
    "tanh":    (np.tanh,                            lambda z: 1 - np.tanh(z) ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)),     lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))),
}

def make_net(sizes, seed=0):
    rng = np.random.default_rng(seed)               # sizes=[1,8,8,1] は 入力1→8→8→出力1
    W = [rng.uniform(-1, 1, (n_out, n_in)) * np.sqrt(3 / n_in) for n_in, n_out in zip(sizes, sizes[1:])]
    B = [np.zeros(n_out) for n_out in sizes[1:]]
    return W, B

def forward(W, B, X, act):                           # X: (データ数, 入力数)
    f, _ = ACTS[act]
    zs, as_ = [], [X]
    for l in range(len(W)):
        z = as_[-1] @ W[l].T + B[l]                  # 層ごとの「ax+b」を全ニューロン分まとめて計算
        zs.append(z)
        as_.append(z if l == len(W) - 1 else f(z))   # 最後の層だけ関数を挟まない
    return zs, as_

def train(W, B, X, T, act, lr=0.02, epochs=20000):
    _, d = ACTS[act]
    for _ in range(epochs):
        zs, as_ = forward(W, B, X, act)
        delta = 2 * (as_[-1] - T) / len(X)           # 誤差が出力にどう効くか
        for l in reversed(range(len(W))):            # 出力側から入力側へ
            gW, gB = delta.T @ as_[l], delta.sum(axis=0)
            if l > 0:
                delta = (delta @ W[l]) * d(zs[l - 1])    # 1つ前の層へ誤差を伝える
            W[l] -= lr * gW
            B[l] -= lr * gB

X = np.linspace(-2, 2, 41).reshape(-1, 1)
T = X ** 2.0

if __name__ == "__main__":
    for sizes, act in [([1, 16, 1], "relu"), ([1, 8, 8, 1], "relu"), ([1, 8, 8, 1], "tanh"), ([1, 8, 1], "sigmoid")]:
        r = []
        for seed in range(3):
            W, B = make_net(sizes, seed)
            train(W, B, X, T, act)
            r.append(float(np.mean((forward(W, B, X, act)[1][-1] - T) ** 2)))
        print(f"{str(sizes):12s} {act:8s} mse = {min(r):.5f} 〜 {max(r):.5f}")
    # 学習範囲の外（x=5）では？
    W, B = make_net([1, 8, 8, 1], 0); train(W, B, X, T, "tanh")
    print("x=1.5 →", forward(W, B, np.array([[1.5]]), "tanh")[1][-1].item(), "(正解 2.25)")
    print("x=5   →", forward(W, B, np.array([[5.0]]), "tanh")[1][-1].item(), "(正解 25)")
