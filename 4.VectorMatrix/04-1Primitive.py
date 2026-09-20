import os

# 0 = |, 1 = -, 2 = /
class TrainingData:
    def __init__(self, directory, filename):
        self.filename = filename
        self.class_id = int(filename[0])
        with open(os.path.join(directory, filename), encoding="utf-8") as f:
            lines = f.read().splitlines()
        if len(lines) != 16 or any(len(line) != 16 for line in lines):
            raise ValueError(filename + " は16文字×16行にしてください")
        self.bits = [1.0 if ch == "■" else 0.0 for line in lines for ch in line]


def predict(bits, weights, bias):
    result = []
    for cls in range(3):
        value = bias[cls]
        for i in range(256):
            value += bits[i] * weights[cls][i]
        result.append(value)
    return result


data_dir = "data3"
training_data = []
for filename in sorted(os.listdir(data_dir)):
    if filename and filename[0] in "012":
        training_data.append(TrainingData(data_dir, filename))

weights = [[0.0 for _ in range(256)] for _ in range(3)]
bias = [0.0, 0.0, 0.0]
rate = 0.04 / 4
epochs = 600

for epoch in range(epochs):
    loss = 0.0
    weight_gradient = [[0.0 for _ in range(256)] for _ in range(3)]
    bias_gradient = [0.0, 0.0, 0.0]

    for td in training_data:
        prediction = predict(td.bits, weights, bias)
        target = [0.0, 0.0, 0.0]
        target[td.class_id] = 1.0

        for cls in range(3):
            diff = prediction[cls] - target[cls]
            loss += diff ** 2
            gradient = 2 * diff
            for i in range(256):
                weight_gradient[cls][i] += gradient * td.bits[i]
            bias_gradient[cls] += gradient

    for cls in range(3):
        for i in range(256):
            weights[cls][i] -= rate * weight_gradient[cls][i]
        bias[cls] -= rate * bias_gradient[cls]

    if epoch % 50 == 0:
        print("epoch =", epoch, "loss =", loss)

print("\n----- result -----")
for td in training_data:
    prediction = predict(td.bits, weights, bias)
    answer = prediction.index(max(prediction))
    print(td.filename, "target =", td.class_id,
          "prediction =", [round(v, 4) for v in prediction], "answer =", answer)
