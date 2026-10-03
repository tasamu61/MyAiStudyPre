public class Main07_1 {

    // 1件のデータ：名前 ＋ 任意個数の諸元
    record Data(String name, double[] values) {

    }

    // 諸元名
    public static void main(String[] args) {
        String[] featureNames = {
            "排気量(cc)",
            "価格(万円)",
            "乗員"
        };

        // 名前 ＋ 諸元
        Data[] data = {
            new Data("軽自動車", new double[]{660, 140, 4}),
            new Data("コンパクト", new double[]{1000, 180, 5}),
            new Data("ヴァン", new double[]{2000, 230, 7}),
            new Data("セダン", new double[]{2000, 240, 5}),
            new Data("スポーツカー", new double[]{2500, 300, 2}),
            new Data("高級車", new double[]{3000, 400, 5})
        };

        Main07_1 main = new Main07_1();
        main.doMain(featureNames, data);
    }

    /**
     * 主処理
     *
     * @param featureNames
     * @param data
     */
    public void doMain(String[] featureNames, Data[] data) {
        int rows = data.length;
        int cols = featureNames.length;

        // ------------------------------------------------------------
        // 元データを行列にする
        // Python:
        // X = np.array([d.values for d in data])
        // ------------------------------------------------------------
        double[][] x = new double[rows][cols];

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                x[row][col] = data[row].values()[col];
            }
        }

        // 1.標準化
        // ------------------------------------------------------------
        // 1-1. 平均を求める
        // Python:
        // mean = X.mean(axis=0)
        // ------------------------------------------------------------
        double[] mean = new double[cols];

        for (int col = 0; col < cols; col++) {
            for (int row = 0; row < rows; row++) {
                mean[col] += x[row][col];
            }
            mean[col] /= rows;
        }

        // ------------------------------------------------------------
        // 1-2. 標準偏差を求める
        // Python:
        // std = X.std(axis=0)
        // ------------------------------------------------------------
        double[] std = new double[cols];

        for (int col = 0; col < cols; col++) {
            for (int row = 0; row < rows; row++) {
                double d = x[row][col] - mean[col];
                std[col] += d * d;
            }
            std[col] = Math.sqrt(std[col] / rows);
        }

        // ------------------------------------------------------------
        // 1-3. 標準化
        // Python:
        // Z = (X - mean) / std
        // ------------------------------------------------------------
        double[][] z = new double[rows][cols];

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                z[row][col]
                        = (x[row][col] - mean[col]) / std[col];
            }
        }

        // ------------------------------------------------------------
        // 標準化した諸元を表示
        // ------------------------------------------------------------
        System.out.println("\n標準化した諸元");
        // Markdown表
        System.out.print("| 名前 |");
        for (String featureName : featureNames) {
            System.out.print(" " + featureName + " |");
        }
        System.out.println();

        System.out.print("| --- |");
        for (int col = 0; col < cols; col++) {
            System.out.print(" ---: |");
        }
        System.out.println();

        for (int row = 0; row < rows; row++) {
            System.out.print("| " + data[row].name() + " |");
            for (int col = 0; col < cols; col++) {
                System.out.printf(" %.4f |", z[row][col]);
            }
            System.out.println();
        }

        // ------------------------------------------------------------
        // 2. 共分散行列
        //　各要素の諸元の関係を表す行列
        // Python:
        // cov = np.cov(Z, rowvar=False, bias=True)
        //
        // 諸元が3個なら 3×3
        // 諸元が5個なら 5×5
        // ------------------------------------------------------------
        double[][] cov = new double[cols][cols];

        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                for (int row = 0; row < rows; row++) {
                    cov[i][j] += z[row][i] * z[row][j];
                }
                cov[i][j] /= rows;
            }
        }

        System.out.println("共分散行列");
        // Markdown表
        System.out.print("|  |");
        for (String featureName : featureNames) {
            System.out.print(" " + featureName + " |");
        }
        System.out.println();

        System.out.print("| --- |");
        for (int col = 0; col < cols; col++) {
            System.out.print(" ---: |");
        }
        System.out.println();

        for (int i = 0; i < cols; i++) {
            System.out.print("| " + featureNames[i] + " |");
            for (int j = 0; j < cols; j++) {
                System.out.printf(" %.4f |", cov[i][j]);
            }
            System.out.println();
        }

        // ------------------------------------------------------------
        // 3. 第1主成分
        //
        // 共分散行列から
        // 「各諸元のX方向への係数」を求める
        //
        // Python:
        // eigenvalues, eigenvectors = np.linalg.eigh(cov)
        // ------------------------------------------------------------
        double[] pc1 = powerMethod(cov);

        System.out.println("\nPC1（X方向）の係数");
        // Markdown表
        System.out.println("| 諸元 | 係数 |");
        System.out.println("| --- | ---: |");

        for (int i = 0; i < cols; i++) {
            System.out.printf(
                    "| %s | %.4f |%n",
                    featureNames[i],
                    pc1[i]
            );
        }

        // 4.第2主成分 PC2 Y方向の係数を求める。
        // 4-1.PC1を共分散行列から取り除くため、PC1の固有値を求める
        double lambda1 = eigenValue(cov, pc1);

        // ------------------------------------------------------------
        // 4-2. PC1を共分散行列から取り除く
        //    → 次に大きな方向を探せるようにする
        // ------------------------------------------------------------
        double[][] cov2 = new double[cols][cols];

        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                cov2[i][j]
                        = cov[i][j]
                        - lambda1 * pc1[i] * pc1[j];
            }
        }

        // ------------------------------------------------------------
        // 4-3. 第2主成分 PC2
        //    新しいY方向の係数
        // ------------------------------------------------------------
        double[] pc2 = powerMethod(cov2);

        System.out.println("\nPC2（Y方向）の係数");
        // Markdown表
        System.out.println("| 諸元 | 係数 |");
        System.out.println("| --- | ---: |");

        for (int i = 0; i < cols; i++) {
            System.out.printf(
                    "| %s | %.4f |%n",
                    featureNames[i],
                    pc2[i]
            );
        }

        // ------------------------------------------------------------
        // 5. 各データをPC1・PC2へ変換
        //
        // Python:
        // W = eigenvectors[:, :2]
        // result = Z @ W
        // ------------------------------------------------------------
        double[][] result = new double[rows][2];

        for (int row = 0; row < rows; row++) {

            for (int col = 0; col < cols; col++) {

                result[row][0]
                        += z[row][col] * pc1[col];

                result[row][1]
                        += z[row][col] * pc2[col];
            }
        }

        System.out.println("\nPCA結果");
        // Markdown表
        System.out.println("| 名前 | PC1 | PC2 |");
        System.out.println("| --- | ---: | ---: |");

        for (int row = 0; row < rows; row++) {
            System.out.printf(
                    "| %s | %.4f | %.4f |%n",
                    data[row].name(),
                    result[row][0],
                    result[row][1]
            );
        }


        // 結果表示

        // ------------------------------------------------------------
        // 元の諸元を表示
        // ------------------------------------------------------------
        System.out.println("\n諸元");
        // Markdown表
        System.out.print("| 名前 |");
        for (String featureName : featureNames) {
            System.out.print(" " + featureName + " |");
        }
        System.out.println();

        System.out.print("| --- |");
        for (int col = 0; col < cols; col++) {
            System.out.print(" ---: |");
        }
        System.out.println();

        for (Data d : data) {
            System.out.print("| " + d.name() + " |");
            for (double value : d.values()) {
                System.out.printf(" %.1f |", value);
            }
            System.out.println();
        }

        // ------------------------------------------------------------
        // PC1をX軸、PC2をY軸としてグラフ表示
        // ------------------------------------------------------------
        printGraph(data, result);
    }

    /**
     * べき乗法
     *
     * @param matrix 共分散行列
     * @return 固有ベクトル // // v(k+1) = A v(k) / |A v(k)| // //
     * 共分散行列から、最も大きな特徴を表す方向を近似する。 // // 3諸元 → 3個の係数 // 5諸元 → 5個の係数 //
     * ===========================================================
     */
    private double[] powerMethod(double[][] matrix) {

        int n = matrix.length;

        double[] v = new double[n];

        // 初期値
        for (int i = 0; i < n; i++) {
            v[i] = 1.0;
        }

        // 繰り返して固有ベクトルへ近づける
        for (int count = 0; count < 100; count++) {

            double[] next = new double[n];

            // next = matrix × v
            // Python:
            // next = matrix @ v
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    next[i] += matrix[i][j] * v[j];
                }
            }

            // ベクトルの長さ
            // Python:
            // norm = np.linalg.norm(next)
            double norm = 0;

            for (int i = 0; i < n; i++) {
                norm += next[i] * next[i];
            }

            norm = Math.sqrt(norm);

            // 長さを1にする
            for (int i = 0; i < n; i++) {
                next[i] /= norm;
            }

            v = next;
        }

        return v;
    }

    /**
     * 固有値を計算する
     *
     * @param matrix 共分散行列
     * @param v 固有ベクトル
     * @return 固有値 // 固有値 λ = v^T A v
     */
    private double eigenValue(
            double[][] matrix,
            double[] v) {

        int n = v.length;

        double value = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                value
                        += v[i] * matrix[i][j] * v[j];
            }
        }

        return value;
    }

    /**
     * 行列を表示する
     *
     * @param matrix
     */
    private void printMatrix(double[][] matrix) {

        for (double[] row : matrix) {
            for (double value : row) {
                System.out.printf("%9.4f ", value);
            }
            System.out.println();
        }
    }

    /**
     * グラフ描画
     *
     * @param data
     * @param result
     */
    private void printGraph(Data[] data, double[][] result) {

        int width = 50;
        int height = 15;

        char[][] graph = new char[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                graph[y][x] = ' ';
            }
        }

        double minX = result[0][0];
        double maxX = result[0][0];
        double minY = result[0][1];
        double maxY = result[0][1];

        for (int i = 1; i < result.length; i++) {
            minX = Math.min(minX, result[i][0]);
            maxX = Math.max(maxX, result[i][0]);
            minY = Math.min(minY, result[i][1]);
            maxY = Math.max(maxY, result[i][1]);
        }

        // PC1, PC2 の値を画面上の座標へ変換
        for (int i = 0; i < result.length; i++) {

            int x = (int) ((result[i][0] - minX)
                    / (maxX - minX)
                    * (width - 1));

            int y = (int) ((result[i][1] - minY)
                    / (maxY - minY)
                    * (height - 1));

            // 配列は上から下なので、Y方向を反転
            y = height - 1 - y;

            graph[y][x] = (char) ('1' + i);
        }

        System.out.println("\nPCAグラフ");
        System.out.println("PC2 ↑");

        for (int y = 0; y < height; y++) {
            System.out.print("|");

            for (int x = 0; x < width; x++) {
                System.out.print(graph[y][x]);
            }

            System.out.println("|");
        }

        System.out.print("+");
        System.out.println("-".repeat(width) + "→ PC1");

        for (int i = 0; i < data.length; i++) {
            System.out.printf(
                    "%d : %s%n",
                    i + 1,
                    data[i].name()
            );
        }
    }
}