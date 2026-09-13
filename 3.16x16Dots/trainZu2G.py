import glob
import os
import random
import sys

# ビット配列と、その解のトレーニングデータ
class TrainingData:
    def __init__(self, label, bits, filename):
        self.label, self.bits, self.filename = label, bits, filename


def _literal_image(rows, x_offset=0, y_offset=3):
    """行の図形を、指定位置に16x16のドット画像として配置する。"""
    image = ["." * 16 for _ in range(16)]
    for index, row in enumerate(rows):
        y = y_offset + index
        left = (16 - len(row)) // 2 + x_offset
        if not 0 <= y < 16 or not 0 <= left <= 16 - len(row):
            raise ValueError("リテラル画像が16x16の範囲外です")
        image[y] = "." * left + row + "." * (16 - left - len(row))
    return image


def _bar_image(width, thickness=1, x_offset=0, y_offset=7):
    """長さ・太さ・位置を変えた横棒画像を作る。"""
    image = ["." * 16 for _ in range(16)]
    left = (16 - width) // 2 + x_offset
    if not 0 <= left <= 16 - width or not 0 <= y_offset <= 16 - thickness:
        raise ValueError("横棒画像が16x16の範囲外です")
    row = "." * left + "■" * width + "." * (16 - left - width)
    for y in range(y_offset, y_offset + thickness):
        image[y] = row
    return image


def _staggered_image(rows, x_offsets, y_offset=5):
    """行ごとに横位置を変えた段違いパターンを作る。"""
    if len(rows) != len(x_offsets):
        raise ValueError("段違いパターンの行数と位置数が違います")
    image = ["." * 16 for _ in range(16)]
    for index, (row, x_offset) in enumerate(zip(rows, x_offsets)):
        y = y_offset + index
        left = (16 - len(row)) // 2 + x_offset
        if not 0 <= y < 16 or not 0 <= left <= 16 - len(row):
            raise ValueError("段違い画像が16x16の範囲外です")
        image[y] = "." * left + row + "." * (16 - left - len(row))
    return image


BAR_TRAINING = [
    ("1_bar_thin_short.txt", _bar_image(4, y_offset=7)),
    ("1_bar_thin_middle.txt", _bar_image(7, y_offset=7)),
    ("1_bar_thin_long.txt", _bar_image(12, y_offset=7)),
    ("1_bar_thick_short.txt", _bar_image(5, thickness=2, y_offset=6)),
    ("1_bar_thick_middle.txt", _bar_image(8, thickness=2, y_offset=6)),
    ("1_bar_thick_long.txt", _bar_image(12, thickness=3, y_offset=6)),
    ("1_bar_upper_left.txt", _bar_image(6, x_offset=-3, y_offset=3)),
    ("1_bar_upper_right.txt", _bar_image(6, x_offset=3, y_offset=3)),
    ("1_bar_lower_left.txt", _bar_image(9, x_offset=-2, y_offset=11)),
    ("1_bar_lower_right.txt", _bar_image(9, x_offset=2, y_offset=11)),
    ("1_bar_staggered_two.txt", _staggered_image(
        ["■■■■■", "■■■■■"], [-2, 2], y_offset=6
    )),
    ("1_bar_staggered_three.txt", _staggered_image(
        ["■■■■", "■■■■", "■■■■"], [-3, 0, 3], y_offset=5
    )),
    ("1_bar_staggered_steps.txt", _staggered_image(
        ["■■■■■■", "■■■■", "■■■■■■"], [-2, 2, -1], y_offset=6
    )),
    ("1_bar_staggered_gap.txt", _staggered_image(
        ["■■■■", "....", "■■■■"], [-2, 2, -2], y_offset=6
    )),
    ("1_bar_staggered_long.txt", _staggered_image(
        ["■■■■■■■■", "....■■■■", "■■■■■■■■"], [-2, 1, -2], y_offset=5
    )),
    ("1_bar_staggered_thick.txt", _staggered_image(
        ["■■■■■■", "■■■■■■", "....", "■■■■■■"], [1, -1, 2, -2], y_offset=5
    )),
]

NOT_BAR_TRAINING = [
    ("0_vertical_short.txt", _literal_image(["■", "■", "■", "■", "■"], x_offset=-1, y_offset=5)),
    ("0_vertical_tall.txt", _literal_image(["■", "■", "■", "■", "■", "■", "■", "■", "■"], x_offset=2, y_offset=3)),
    ("0_diagonal_down.txt", _literal_image(["■.......", ".■......", "..■.....", "...■....", "....■...", ".....■..", "......■."], y_offset=4)),
    ("0_diagonal_up.txt", _literal_image(["......■.", ".....■..", "....■...", "...■....", "..■.....", ".■......", "■......."], y_offset=4)),
    ("0_l_shape.txt", _literal_image(["■.......", "■.......", "■.......", "■.......", "■■■■■■.."], y_offset=5)),
    ("0_cross.txt", _literal_image(["...■...", "...■...", "■■■■■■■", "...■...", "...■..."], y_offset=5)),
    ("0_square.txt", _literal_image(["■■■■■", "■...■", "■...■", "■■■■■"], y_offset=5)),
    ("0_zigzag.txt", _literal_image(["■■....", "..■■..", "....■■", "..■■..", "■■...."], y_offset=5)),
    ("0_v_shape.txt", _literal_image(["■.......", ".■......", "..■.....", "...■....", "..■.....", ".■......", "■......."], y_offset=4)),
    ("0_t_shape.txt", _literal_image(["■■■■■■■", "...■...", "...■...", "...■...", "...■..."], y_offset=4)),
]

BAR_EVALUATION = [
    ("1_bar_eval_short.txt", _bar_image(6, y_offset=5)),
    ("1_bar_eval_thick.txt", _bar_image(10, thickness=2, x_offset=-1, y_offset=8)),
    ("1_bar_eval_long.txt", _bar_image(14, y_offset=10)),
    ("0_not_bar_eval_cross.txt", _literal_image(["..■..", "..■..", "■■■■■", "..■..", "..■.."], y_offset=5)),
    ("0_not_bar_eval_diagonal.txt", _literal_image(["■.......", ".■......", "..■.....", "...■....", "....■...", ".....■..", "......■."], y_offset=4)),
    ("1_bar_eval_staggered.txt", _staggered_image(
        ["■■■■■", "■■■■■"], [-2, 2], y_offset=6
    )),
    ("1_bar_eval_steps.txt", _staggered_image(
        ["■■■■", "■■■■", "■■■■"], [-3, 0, 3], y_offset=5
    )),
    ("0_not_bar_eval_split.txt", _staggered_image(
        ["■■■■", "....", "■■■■"], [2, -2, 2], y_offset=6
    )),
]


TWO_TRAINING = [
    ("1_2_standard.txt", _literal_image([
        "■■■■■■", "......■", "......■", "■■■■■■", "■.......", "■.......", "■■■■■■"
    ])),
    ("1_2_shifted.txt", _literal_image([
        "■■■■■■■", "......■", ".....■.", "...■■..", "..■....", "■......", "■■■■■■■"
    ])),
    ("1_2_thin.txt", _literal_image([
        "■■■■■", "....■", "...■.", "..■..", ".■...", "■....", "■■■■■"
    ])),
    ("1_2_wide.txt", _literal_image([
        "■■■■■■■■", ".......■", ".......■", "■■■■■■■■", "■.......", "■.......", "■■■■■■■■"
    ])),
    ("1_2_broken.txt", _literal_image([
        "■■■.■■■", "......■", "......■", "■■■■■■■", "■.......", "■.......", "■■■■■■■"
    ])),
    ("1_2_left.txt", _literal_image([
        "■■■■■■", ".....■.", "....■..", "...■...", "..■....", ".■.....", "■■■■■■"
    ])),
    ("1_2_right.txt", _literal_image([
        "■■■■■■", "......■", "......■", "...■■■.", "..■....", ".■.....", "■■■■■■"
    ])),
    ("1_2_round.txt", _literal_image([
        ".■■■■..", "■....■.", ".....■.", "...■■..", "..■....", "■......", "■■■■■."
    ])),
    ("1_2_thick.txt", _literal_image([
        "■■■■■■", "■■...■", "....■■", "■■■■■■", "■■....", "■.....", "■■■■■■"
    ])),
    ("1_2_short.txt", _literal_image([
        "■■■■", "...■", "..■.", ".■..", "■...", "■■■■"
    ])),
    ("1_2_tall.txt", _literal_image([
        "■■■■■■", ".....■", ".....■", "...■■.", "..■...", ".■....", "■.....", "■.....", "■■■■■■"
    ], y_offset=2)),
    ("1_2_left_shift.txt", _literal_image([
        "■■■■■", "....■", "...■.", "..■..", ".■...", "■....", "■■■■■"
    ], x_offset=-2, y_offset=4)),
    ("1_2_right_shift.txt", _literal_image([
        "■■■■■■■", "......■", ".....■.", "....■..", "...■...", "..■....", "■■■■■■■"
    ], x_offset=1, y_offset=1)),
    ("1_2_open_curve.txt", _literal_image([
        ".■■■■.", "■....■", ".....■", "....■.", "...■..", "..■...", "■■■■■■"
    ], x_offset=-1, y_offset=5)),
    ("1_2_narrow_tall.txt", _literal_image([
        "■■■", "..■", ".■.", "■..", "■..", "■■■"
    ], x_offset=2, y_offset=6)),
    ("1_2_block.txt", _literal_image([
        "■■■■■■■■", "■■■■■■■■", "......■■", "....■■..", "..■■....", "■■......", "■■■■■■■■", "■■■■■■■■"
    ], x_offset=-1, y_offset=1)),
    ("1_2_round_wide.txt", _literal_image([
        ".■■■■■■.", "■■....■■", "......■■", "....■■..", "..■■....", "■■......", "■■■■■■■■"
    ], x_offset=1, y_offset=3)),
    ("1_2_handwritten.txt", _literal_image([
        "..■■■■..", ".■....■.", "......■.", ".....■..", "...■■...", ".■■.....", "■■■■■■.."
    ], x_offset=-2, y_offset=4)),
    ("1_2_angular.txt", _literal_image([
        "■■■■■■■", ".....■■", "....■■.", "...■■..", "..■■...", ".■■....", "■■■■■■■"
    ], x_offset=2, y_offset=2)),
    ("1_2_low_profile.txt", _literal_image([
        "■■■■■■■■■■", "........■.", ".......■..", "......■...", ".....■....", "....■■■■■■"
    ], x_offset=-2, y_offset=7)),
]

NOT_TWO_TRAINING = [
    ("0_3.txt", _literal_image(["■■■■■■", "......■", "......■", "■■■■■■", "......■", "......■", "■■■■■■"])),
    ("0_5.txt", _literal_image(["■■■■■■", "■.......", "■.......", "■■■■■■", "......■", "......■", "■■■■■■"])),
    ("0_7.txt", _literal_image(["■■■■■■■", "......■", ".....■.", "....■..", "...■...", "...■...", "...■..."])),
    ("0_8.txt", _literal_image(["■■■■■■", "■......■", "■......■", "■■■■■■", "■......■", "■......■", "■■■■■■"])),
    ("0_9.txt", _literal_image(["■■■■■■", "■......■", "■......■", "■■■■■■■", "......■", "......■", "■■■■■■"])),
    ("0_1.txt", _literal_image(["...■...", "..■■...", "...■...", "...■...", "...■...", "...■...", "■■■■■■"])),
    ("0_4.txt", _literal_image(["■....■", "■....■", "■....■", "■■■■■■", ".....■", ".....■", ".....■"])),
    ("0_6.txt", _literal_image(["..■■■■", ".■....", "■.....", "■■■■■.", "■....■", "■....■", ".■■■■."])),
    ("0_0.txt", _literal_image([".■■■■.", "■....■", "■....■", "■....■", "■....■", "■....■", ".■■■■."])),
    ("0_3_thin.txt", _literal_image(["■■■■■", "....■", "....■", ".■■■■", "....■", "....■", "■■■■■"])),
    ("0_4_tall.txt", _literal_image(["■..■", "■..■", "■..■", "■■■■", "...■", "...■", "...■", "...■", "...■"], y_offset=2)),
    ("0_6_left.txt", _literal_image([".■■■■", "■....", "■....", ".■■■.", "■..■.", "■..■.", ".■■.."], x_offset=-2, y_offset=4)),
    ("0_8_shifted.txt", _literal_image([".■■■■.", "■....■", "■....■", ".■■■■.", "■....■", "■....■", ".■■■■."], x_offset=1, y_offset=1)),
    ("0_9_tall.txt", _literal_image([".■■■■.", "■....■", "■....■", ".■■■■■", ".....■", ".....■", ".■■■■."], y_offset=2)),
    ("0_1_short.txt", _literal_image(["..■..", ".■■..", "..■..", "..■..", ".■■■."], x_offset=-1, y_offset=6)),
    ("0_3_round.txt", _literal_image([
        ".■■■■■.", "■.....■", "......■", ".■■■■■.", "......■", "■.....■", ".■■■■■."
    ], x_offset=1, y_offset=3)),
    ("0_5_angular.txt", _literal_image([
        "■■■■■■", "■......", "■......", "■■■■■.", ".....■", ".....■", "■■■■■."
    ], x_offset=-2, y_offset=4)),
    ("0_8_thick.txt", _literal_image([
        ".■■■■.", "■■■■■■", "■■..■■", "■■■■■■", "■■..■■", "■■■■■■", ".■■■■."
    ], x_offset=1, y_offset=2)),
    ("0_9_loop.txt", _literal_image([
        ".■■■■.", "■....■", "■....■", ".■■■■■", ".....■", "■....■", ".■■■■."
    ], x_offset=-1, y_offset=3)),
    ("0_7_diagonal.txt", _literal_image([
        "■■■■■■■■", "......■■", ".....■■.", "....■■..", "...■■...", "..■■....", ".■■....."
    ], x_offset=-1, y_offset=5)),
]

TWO_EVALUATION = [
    ("1_2_eval_a.txt", _literal_image(["■■■■■■", ".....■.", ".....■.", "■■■■■■", "■.......", "■.......", "■■■■■■"])),
    ("1_2_eval_b.txt", _literal_image(["■■■■■■■", "......■", ".....■.", "....■..", "...■...", "..■....", "■■■■■■■"])),
    ("0_3_eval.txt", _literal_image(["■■■■■■", "......■", "......■", "■■■■■■", "......■", "......■", "■■■■■■"])),
    ("0_5_eval.txt", _literal_image(["■■■■■■", "■.......", "■.......", "■■■■■■", "......■", "......■", "■■■■■■"])),
    ("0_7_eval.txt", _literal_image(["■■■■■■", ".....■.", "....■..", "...■...", "...■...", "...■...", "...■..."])),
]


def load_literal_data(records):
    return [
        TrainingData(name.startswith("1_"), [[ch == "■" for ch in row] for row in image], name)
        for name, image in records
    ]

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

# 判定対象のビット配列(bits)と学習済みの重み配列(weights)を演算し、
# 学習で補正された基準値(bias)を加えて予測値を求める
def predict(bits, weights, bias):
    prediction = bias
    for y in range(16):
        for x in range(16):
            prediction += (1.0 if bits[y][x] else 0.0) * weights[y][x]
    return prediction

def load_training_data(folder):
    """指定フォルダの16x16データを読み込む。"""
    return [load_TrainingData(filename) for filename in sorted(
        glob.glob(os.path.join(folder, "*.txt"))
    )]


def create_model():
    """未学習の重みとbiasを作る。"""
    return [[0.0] * 16 for _ in range(16)], 0.0


def train_one_epoch(training_data, weights, bias, learning_rate=0.01):
    """1 epochだけ学習し、更新後のLossとbiasを返す。"""
    total_loss = 0.0
    shuffled_data = list(training_data)
    random.shuffle(shuffled_data)

    for s in shuffled_data:
        prediction = predict(s.bits, weights, bias)   # ①予測

        # 数字2なら target=1、数字2でなければ target=0
        target = 1.0 if s.label else 0.0

        # ②Loss
        loss = (prediction - target) ** 2
        total_loss += loss

        # ③Lossを微分
        # Loss = (prediction - target)²
        # 微分 = 2 * (prediction - target)
        gradient = 2 * (prediction - target)

        # ④256個のweightを補正
        # ONであるマスのweightを補正する
        for y in range(16):
            for x in range(16):
                input_value = 1.0 if s.bits[y][x] else 0.0
                weights[y][x] -= learning_rate * gradient * input_value

        # ⑤biasを補正
        bias -= learning_rate * gradient

    return total_loss / len(training_data), bias


def evaluate(training_data, weights, bias):
    """各データの予測値、正解ラベル、判定結果を返す。"""
    results = []
    for data in training_data:
        prediction = predict(data.bits, weights, bias)
        results.append({
            "filename": data.filename,
            "prediction": prediction,
            "label": data.label,
            "answer": prediction >= 0.5,
        })
    return results


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    training_data = load_literal_data(BAR_TRAINING + NOT_BAR_TRAINING)
    weights, bias = create_model()

    print("===== 学習前 =====")
    for result in evaluate(training_data, weights, bias):
        print(result["filename"], "教師=", result["label"],
              "非合致度=", round(result["prediction"], 4))

    epochs = 1000
    for epoch in range(epochs):
        loss, bias = train_one_epoch(training_data, weights, bias)
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print("epoch =", epoch + 1, "loss =", round(loss, 6))

    print("\n===== 学習後 =====")
    for result in evaluate(training_data, weights, bias):
        print(result["filename"], "教師値=", result["label"],
              "非合致度=", round(result["prediction"], 4))

    if len(sys.argv) >= 2:
        test_data = load_TrainingData(sys.argv[1])

        print("\n===== 評価画像 =====")
        for row in test_data.bits:
            print("".join("■" if v else " " for v in row))

        p = predict(test_data.bits, weights, bias)

        print("\nファイル =", sys.argv[1])
        print("非合致度 =", round(p, 4))

        print("判定 =", "数字2" if p >= 0.5 else "数字2ではない")


if __name__ == "__main__":
    main()