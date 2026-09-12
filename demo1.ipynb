import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# =================================================================
# 1. データ生成ロジック
# =================================================================
def generate_data(dataset_type, has_outliers):
    np.random.seed(42)  # 再現性のためのシード固定
    
    if dataset_type == 'fixed':
        x = np.array([1, 3, 3, 7], dtype=float)
        y = np.array([5, 11, 9, 19], dtype=float)
    elif dataset_type == 'linear_noise':
        x = np.linspace(-5, 5, 50)
        y = 2 * x + 1 + np.random.normal(0, 1.5, size=50)
    elif dataset_type == 'quad_noise':
        x = np.linspace(-3, 3, 50)
        y = 0.8 * (x ** 2) - 1 * x - 2 + np.random.normal(0, 1.0, size=50)

    if has_outliers and dataset_type != 'fixed':
        x_outliers = np.array([-2, 0, 2], dtype=float)
        y_outliers = np.array([15, -15, 18], dtype=float)
        x = np.concatenate([x, x_outliers])
        y = np.concatenate([y, y_outliers])

    return x, y

# =================================================================
# 2. 計算ロジック (予測・Loss・勾配降下)
# =================================================================
def predict(x, model_degree, params):
    if model_degree == 1:
        return params['a'] * x + params['b']
    elif model_degree == 2:
        return params['a'] * (x ** 2) + params['b'] * x + params['c']
    elif model_degree == 3:
        return params['a'] * (x ** 3) + params['b'] * (x ** 2) + params['c'] * x + params['d']

def calculate_loss(x, y, model_degree, params, loss_type):
    pred = predict(x, model_degree, params)
    err = pred - y
    if loss_type == 'mse':
        return np.mean(err ** 2)
    elif loss_type == 'mae':
        return np.mean(np.abs(err))

def step_gradient(x, y, model_degree, params, loss_type, lr):
    n = len(x)
    pred = predict(x, model_degree, params)
    err = pred - y

    if loss_type == 'mse':
        d_err = 2 * err
    elif loss_type == 'mae':
        d_err = np.sign(err)

    if model_degree == 1:
        grad_a = np.sum(d_err * x) / n
        grad_b = np.sum(d_err * 1) / n
        params['a'] -= lr * grad_a
        params['b'] -= lr * grad_b
    elif model_degree == 2:
        grad_a = np.sum(d_err * (x ** 2)) / n
        grad_b = np.sum(d_err * x) / n
        grad_c = np.sum(d_err * 1) / n
        params['a'] -= lr * grad_a
        params['b'] -= lr * grad_b
        params['c'] -= lr * grad_c
    elif model_degree == 3:
        grad_a = np.sum(d_err * (x ** 3)) / n
        grad_b = np.sum(d_err * (x ** 2)) / n
        grad_c = np.sum(d_err * x) / n
        grad_d = np.sum(d_err * 1) / n
        params['a'] -= lr * grad_a
        params['b'] -= lr * grad_b
        params['c'] -= lr * grad_c
        params['d'] -= lr * grad_d

# =================================================================
# 3. UIと状態管理 (Colab/Jupyterインタラクティブ画面)
# =================================================================
class RegressionApp:
    def __init__(self):
        # 状態変数
        self.epoch = 0
        self.params = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}

        # UI要素の定義
        self.dataset_select = widgets.Dropdown(
            options=[('一次の固定直線 (4点)', 'fixed'), 
                     ('一次 50点 (ばらつきあり)', 'linear_noise'), 
                     ('二次 50点 (ばらつきあり)', 'quad_noise')],
            value='linear_noise',
            description='データセット:'
        )
        self.outlier_check = widgets.Checkbox(value=False, description='外れ値(3点)を追加')
        self.model_select = widgets.Dropdown(
            options=[('一次式: y = a*x + b', 1), 
                     ('二次式: y = a*x² + b*x + c', 2), 
                     ('三次式: y = a*x³ + b*x² + c*x + d', 3)],
            value=1,
            description='モデル:'
        )
        self.loss_select = widgets.Dropdown(
            options=[('MSE (平均二乗誤差)', 'mse'), ('MAE (平均絶対誤差)', 'mae')],
            value='mse',
            description='損失関数:'
        )
        
        self.init_a = widgets.FloatText(value=0.0, description='初期値 a:')
        self.init_b = widgets.FloatText(value=0.0, description='初期値 b:')
        self.init_c = widgets.FloatText(value=0.0, description='初期値 c:')
        self.init_d = widgets.FloatText(value=0.0, description='初期値 d:')
        
        self.lr_input = widgets.FloatText(value=0.01, description='学習率:')
        
        # ボタン群
        self.btn_1 = widgets.Button(description='+1 Epoch', button_style='info')
        self.btn_5 = widgets.Button(description='+5 Epoch', button_style='info')
        self.btn_10 = widgets.Button(description='+10 Epoch', button_style='info')
        self.btn_100 = widgets.Button(description='+100 Epoch', button_style='info')
        self.btn_reset = widgets.Button(description='リセット', button_style='warning')
        
        self.out = widgets.Output()

        # 【修正点】まずデータを最初に生成しておく
        self.x_data, self.y_data = generate_data(self.dataset_select.value, self.outlier_check.value)

        # イベント接続
        self.dataset_select.observe(self.on_config_change, names='value')
        self.outlier_check.observe(self.on_config_change, names='value')
        self.model_select.observe(self.on_model_change, names='value')
        self.loss_select.observe(self.render, names='value')
        
        for widget in [self.init_a, self.init_b, self.init_c, self.init_d]:
            widget.observe(self.reset_params, names='value')

        self.btn_1.on_click(lambda b: self.train(1))
        self.btn_5.on_click(lambda b: self.train(5))
        self.btn_10.on_click(lambda b: self.train(10))
        self.btn_100.on_click(lambda b: self.train(100))
        self.btn_reset.on_click(lambda b: self.reset_params())

        # 初期描画
        self.on_model_change(None)

    def on_model_change(self, change):
        degree = self.model_select.value
        # モデルごとの適正学習率の初期設定
        if degree == 1:
            self.lr_input.value = 0.01
            self.init_c.layout.display = 'none'
            self.init_d.layout.display = 'none'
        elif degree == 2:
            self.lr_input.value = 0.0001
            self.init_c.layout.display = 'flex'
            self.init_d.layout.display = 'none'
        elif degree == 3:
            self.lr_input.value = 0.000001
            self.init_c.layout.display = 'flex'
            self.init_d.layout.display = 'flex'
        self.reset_params()

    def on_config_change(self, change):
        self.x_data, self.y_data = generate_data(self.dataset_select.value, self.outlier_check.value)
        self.reset_params()

    def reset_params(self, change=None):
        self.params = {
            'a': float(self.init_a.value),
            'b': float(self.init_b.value),
            'c': float(self.init_c.value),
            'd': float(self.init_d.value)
        }
        self.epoch = 0
        self.render()

    def train(self, count):
        degree = self.model_select.value
        loss_type = self.loss_select.value
        lr = self.lr_input.value

        for _ in range(count):
            step_gradient(self.x_data, self.y_data, degree, self.params, loss_type, lr)
            self.epoch += 1
        self.render()

    def render(self, change=None):
        # 安全対策：データが空の場合は描画しない
        if len(self.x_data) == 0:
            return

        with self.out:
            clear_output(wait=True)
            degree = self.model_select.value
            loss_type = self.loss_select.value

            fig, ax = plt.subplots(figsize=(7, 5))
            
            # 散布図プロット
            ax.scatter(self.x_data, self.y_data, color='red', label='Data Points', zorder=3)
            
            # 予測曲線のプロット
            x_min, x_max = np.min(self.x_data) - 1, np.max(self.x_data) + 1
            x_line = np.linspace(x_min, x_max, 200)
            y_line = predict(x_line, degree, self.params)
            
            ax.plot(x_line, y_line, color='dodgerblue', linewidth=2, label='Prediction')
            
            # 軸と範囲の設定
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_xlim(x_min, x_max)
            y_margin = (np.max(self.y_data) - np.min(self.y_data)) * 0.2
            ax.set_ylim(np.min(self.y_data) - y_margin, np.max(self.y_data) + y_margin)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
            plt.show()

            # ステータス表示
            current_loss = calculate_loss(self.x_data, self.y_data, degree, self.params, loss_type)
            
            param_str = f"a={self.params['a']:.4f}, b={self.params['b']:.4f}"
            if degree >= 2:
                param_str += f", c={self.params['c']:.4f}"
            if degree == 3:
                param_str += f", d={self.params['d']:.4f}"

            loss_str = f"{current_loss:.4f}" if np.isfinite(current_loss) else "発散 (学習率が高すぎます)"
            
            print(f"=== 学習ステータス ===")
            print(f"Epoch     : {self.epoch}")
            print(f"Loss ({loss_type.upper()}) : {loss_str}")
            print(f"Parameters: {param_str}")

    def display_ui(self):
        # UIレイアウト作成
        box_data = widgets.VBox([self.dataset_select, self.outlier_check, self.model_select, self.loss_select])
        box_params = widgets.VBox([self.init_a, self.init_b, self.init_c, self.init_d, self.lr_input])
        box_buttons = widgets.HBox([self.btn_1, self.btn_5, self.btn_10, self.btn_100, self.btn_reset])
        
        controls = widgets.VBox([box_data, widgets.HTML("<hr><b>初期値・学習率:</b>"), box_params, widgets.HTML("<hr>"), box_buttons])
        ui = widgets.HBox([controls, self.out])
        display(ui)

# アプリケーションの起動
app = RegressionApp()
app.display_ui()
