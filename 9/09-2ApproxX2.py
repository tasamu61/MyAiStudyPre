# 09-2ApproxX2.py ― y = x**2 を「y=ax+b と ReLU」で近似する（素のPython、Colabでそのまま実行可）
import random

random.seed(1)

# =====================================================
# A. 部品：直線、ReLU、ニューロン1個
# =====================================================
def linear(x, a, b):
    return a * x + b          # 1章の y = a*x + b そのもの

def relu(z):
    return max(0.0, z)        # 負なら0、正ならそのまま

def neuron(x, w, b):
    return relu(linear(x, w, b))   # 直線 → ReLU

# =====================================================
# B. ネットワーク：ニューロンを H 個並べて、倍率 V をかけて足す
# =====================================================
def predict(net, x):
    W, B, V, C = net["W"], net["B"], net["V"], net["C"]
    y = C
    for i in range(len(W)):
        y += V[i] * neuron(x, W[i], B[i])
    return y

def mse(net, data):
    return sum((predict(net, x) - t) ** 2 for x, t in data) / len(data)

# 学習データ：y = x**2 の 41点
data = [(-2 + i * 0.1, (-2 + i * 0.1) ** 2.0) for i in range(41)]

# =====================================================
# C. 手で作った重み（5点が x**2 と一致する）
# =====================================================
def part_hand():
    hand = {"W": [1, 1, 1, 1], "B": [2, 1, 0, -1], "V": [-3, 2, 2, 2], "C": 4}
    print("【手作りの重み】 x, 正解, 予測")
    for x in [-2, -1, 0, 1, 2]:
        print(f"  {x:3d} {x**2:6.2f} {predict(hand, x):6.2f}")

# =====================================================
# D. 学習：誤差が減る方向に W, B, V, C を動かす
# =====================================================
def make_net(H):
    W = [random.choice([-1.0, 1.0]) for _ in range(H)]
    knots = [-2 + (i + 0.5) * (4 / H) for i in range(H)]     # 折れ目を -2〜2 に散らす
    B = [-W[i] * knots[i] for i in range(H)]
    V = [(random.random() - 0.5) * 0.6 for _ in range(H)]
    return {"W": W, "B": B, "V": V, "C": 0.0}

def train(net, data, lr, epochs, report=()):
    W, B, V = net["W"], net["B"], net["V"]
    H, N = len(W), len(data)
    for epoch in range(epochs + 1):
        if epoch in report:
            show(net, epoch)
        gW, gB, gV, gC = [0.0] * H, [0.0] * H, [0.0] * H, 0.0
        for x, t in data:
            dy = 2 * (predict(net, x) - t) / N            # 誤差が y にどう効くか
            gC += dy
            for i in range(H):
                z = W[i] * x + B[i]
                if z > 0:                                  # ReLU が働いているときだけ
                    gV[i] += dy * z
                    gW[i] += dy * V[i] * x
                    gB[i] += dy * V[i]
        for i in range(H):
            W[i] -= lr * gW[i]; B[i] -= lr * gB[i]; V[i] -= lr * gV[i]
        net["C"] -= lr * gC

def show(net, epoch):
    ys = [predict(net, x) for x in (-2, -1, 0, 1, 2)]
    knots = ", ".join(f"{-b / w:5.2f}" for w, b in zip(net["W"], net["B"]))
    print(f"  {epoch:6d}回目 mse={mse(net, data):8.5f}  y(-2..2)=" +
          " ".join(f"{y:5.2f}" for y in ys) + f"  折れ目x=[{knots}]")

def part_train():
    print("【学習の過程】 H=4（正解 y(-2..2)= 4.00 1.00 0.00 1.00 4.00）")
    train(make_net(4), data, lr=0.02, epochs=20000, report=(0, 100, 1000, 5000, 20000))

# =====================================================
# E. 横に増やす（ルートの数 H）
# =====================================================
def part_width():
    print("【横：ニューロン数 H と最終誤差（3回試行の最小〜最大）】")
    for H in [1, 2, 4, 16]:
        r = []
        for _ in range(3):
            net = make_net(H); train(net, data, lr=0.02, epochs=10000); r.append(mse(net, data))
        print(f"  H={H:2d}  {min(r):.5f} 〜 {max(r):.5f}")

# =====================================================
# F. 縦に重ねる（山型を何回も重ねると折れ目が倍々に増える）
# =====================================================
def tent(x):
    return 2 * relu(x) - 4 * relu(x - 0.5)   # ニューロン2個

def pieces(f, n=4096):
    cnt, prev = 1, None
    for i in range(n):
        s = (f((i + 1) / n) - f(i / n)) * n
        if prev is not None and abs(s - prev) > 1e-6:
            cnt += 1
        prev = s
    return cnt

def part_depth():
    print("【縦：山型を重ねる】")
    f = lambda x: x
    for k in range(1, 6):
        g = f
        f = lambda x, g=g: tent(g(x))
        print(f"  {k}層 ニューロン{2*k:2d}個 → 折れ線の区間数 {pieces(f)}")

# =====================================================
# G. 間に挟む関数
# =====================================================
import math
acts = {
    "relu":    lambda z: max(0.0, z),
    "leaky":   lambda z: z if z > 0 else 0.1 * z,
    "tanh":    math.tanh,
    "sigmoid": lambda z: 1 / (1 + math.exp(-z)),
    "linear":  lambda z: z,
}
def part_acts():
    zs = [-2, -1, 0, 1, 2]
    print("【関数の値】 z=" + " ".join(f"{z:6d}" for z in zs))
    for name, f in acts.items():
        print(f"  {name:8s} " + " ".join(f"{f(z):6.2f}" for z in zs))
    e = [math.exp(z) for z in (2.0, 1.0, 0.1)]
    print("  softmax([2.0,1.0,0.1]) =", [round(v / sum(e), 2) for v in e])

if __name__ == "__main__":
    part_hand(); part_acts(); part_depth(); part_train(); part_width()
