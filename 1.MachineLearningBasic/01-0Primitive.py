# 学習データ
training_data = [
    (2.0,  4.0),
    (4.0,  8.0),
    (6.0, 12.0),
    (8.0, 16.0)
]

# 初期値
weight = 0.5
bias = 0.0

# 学習率
learning_rate = 0.03

# 学習回数
epochs = 100000



for epoch in range(epochs):

    total_loss = 0.0

    for x, target in training_data:

        # 1. 予測
        prediction = x * weight + bias

        # 2. 正解との誤差（Loss）
        loss = (prediction - target) ** 2
        total_loss += loss

        # 3. Lossをweightについて微分
        weight_gradient = 2 * (prediction - target) * x

        # 4. Lossをbiasについて微分
        bias_gradient = 2 * (prediction - target)

        # 5. weightとbiasを補正
        weight = weight - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(
            f"{epoch + 1:3d}回目 "
            f"loss={total_loss:.6f} "
            f"weight={weight:.6f} "
            f"bias={bias:.6f}"
        )


# 学習結果
print()
print("学習後のweight =", weight)
print("学習後のbias   =", bias)


# 問題
x = 10.0

# 推論
prediction = x * weight + bias

print()
print("問題 =", x)
print("予測 =", prediction)
