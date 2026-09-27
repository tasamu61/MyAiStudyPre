import java.util.Arrays;
import java.util.List;

/**
 * データ（名前 + 可変長の諸元）を主成分分析（PCA）によって
 * 3次元以上 → 2次元 へ次元削減し、結果を表示するプログラム（1ファイル完結版）。
 *
 * 【データを差し替えたい場合】
 * 内部クラス DataSet を継承した新しいクラス（CarDataSet を参照）を作成し、
 * main() 内の "new CarDataSet()" の部分を差し替えるだけでよい。
 * 諸元の数(m)・データの件数(n)がいくつであっても、以下のアルゴリズムは自動的に対応する。
 *
 * 【次元削減（PCA）の流れ】
 * Step0  : 各諸元（元データ）の確認
 * Step1  : 標準化      … 各変数のスケール（単位）の違いをなくし、平均0・標準偏差1に揃える
 * Step2  : 共分散行列の計算 … 変数間の関係性（相関）を数値化する
 * Step3  : 第1主成分の係数を求める … 共分散行列を固有値分解し、固有値が最大の固有ベクトルを採用
 * Step4  : 第2主成分の係数を求める … 同じ固有値分解の結果から、固有値が2番目に大きい固有ベクトルを採用
 * Step5  : 各データをPC1・PC2へ変換 … 標準化データを2つの主成分軸に射影＝次元削減の本体
 * (参考) : 寄与率の算出（結果の評価用の補足情報。次元削減の本質的な処理ではない）
 * FINAL  : 結果の可視化（テキスト散布図）
 */
public class PCA2D {

    // ===================== データ定義（内部クラス） =====================

    /**
     * 1件分のデータを表すレコード。
     * name   : データの名前（例: 車種名）
     * values : 諸元の値。要素数は自由（諸元が増えても減っても対応できる）。
     */
    record Data(String name, double[] values) {
    }

    /**
     * PCAにかけるデータセットを表す抽象クラス。
     * 別のデータを分析したい場合は、このクラスを継承した新しいクラス
     * （例: BikeDataSet）を作成し、featureNames() と records() を実装するだけでよい。
     */
    static abstract class DataSet {
        /** 諸元（列）の名前一覧。要素数がそのまま諸元の数(m)になる。 */
        abstract String[] featureNames();

        /** データ本体（行）の一覧。件数(n)は自由に増減できる。 */
        abstract List<Data> records();
    }

    /**
     * サンプルデータセット：クルマの諸元。
     * 諸元（列）を増やしたい場合は featureNames() に名前を追加し、
     * 各 records() の values 配列にも同じ順番で値を追加すればよい。
     * データ（行）を増やしたい場合は records() に Data を追加するだけでよい。
     */
    static class CarDataSet extends DataSet {
        @Override
        String[] featureNames() {
            return new String[]{"排気量(cc)", "価格(万円)", "乗員"};
        }

        @Override
        List<Data> records() {
            return List.of(
                    new Data("軽自動車",     new double[]{660,  140, 4}),
                    new Data("コンパクト",   new double[]{1000, 180, 5}),
                    new Data("ヴァン",       new double[]{2000, 230, 7}),
                    new Data("セダン",       new double[]{2000, 240, 5}),
                    new Data("スポーツカー", new double[]{2500, 300, 2}),
                    new Data("高級車",       new double[]{3000, 400, 5})
            );
        }
    }

    // ===================== メイン処理 =====================

    public static void main(String[] args) {
        // ここを別の DataSet 実装に差し替えるだけで、別データの分析に切り替えられる。
        DataSet dataset = new CarDataSet();
        run(dataset);
    }

    /** PCAの本処理。DataSetの中身（諸元数・件数）が変わってもそのまま動く。 */
    static void run(DataSet dataset) {
        String[] features = dataset.featureNames();
        List<Data> records = dataset.records();

        int n = records.size();
        int m = features.length;

        if (n < 2) {
            throw new IllegalArgumentException("データは2件以上必要です。");
        }
        if (m < 2) {
            throw new IllegalArgumentException("2次元に削減するには諸元が2つ以上必要です。");
        }
        // 全レコードの諸元数が featureNames() の数と一致しているかを検証する
        for (Data r : records) {
            if (r.values().length != m) {
                throw new IllegalArgumentException(
                        "「" + r.name() + "」の諸元数(" + r.values().length
                                + ")が featureNames() の数(" + m + ")と一致しません。");
            }
        }

        String[] names = new String[n];
        double[][] data = new double[n][m];
        for (int i = 0; i < n; i++) {
            names[i] = records.get(i).name();
            data[i] = records.get(i).values();
        }

        System.out.println("# 二次元変換（主成分分析: PCA）処理結果\n");
        System.out.println("（データ件数: " + n + " 件, 諸元数: " + m + " 個）\n");

        // ---- Step0: 各諸元（元データ） ----
        printTable("Step0: 各諸元（元データ）", names, features, data);

        // ---- Step1: 標準化 ----
        // 諸元ごとに単位・スケールが異なるため、(値-平均)/標準偏差 で
        // 平均0・標準偏差1に揃える。諸元数がいくつでもループで自動処理される。
        double[] mean = new double[m];
        double[] std = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += data[i][j];
            mean[j] = sum / n;
        }
        for (int j = 0; j < m; j++) {
            double sq = 0;
            for (int i = 0; i < n; i++) sq += Math.pow(data[i][j] - mean[j], 2);
            std[j] = Math.sqrt(sq / n); // 母標準偏差（n で除算）を採用
        }
        double[][] z = new double[n][m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                z[i][j] = (data[i][j] - mean[j]) / std[j];

        printMeanStd(features, mean, std);
        printTable("Step1: 標準化後データ (z-score)", names, features, z);

        // ---- Step2: 共分散行列の計算 ----
        // 標準化後データの共分散行列は、対角成分が各変数の分散(=1)、
        // 非対角成分が変数間の相関係数と一致する（＝相関行列と等価）。
        // m×m の行列になるため、諸元が増えても自動的にサイズが変わる。
        double[][] cov = new double[m][m];
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                double s = 0;
                for (int i = 0; i < n; i++) s += z[i][a] * z[i][b];
                cov[a][b] = s / n;
            }
        }
        printMatrix("Step2: 共分散行列（標準化後のため相関行列と一致）", features, cov);

        // ---- Step3・Step4の準備: 固有値・固有ベクトルの計算（ヤコビ法） ----
        // 共分散行列を対角化することで、分散を最大にする方向（固有ベクトル）と
        // その分散の大きさ（固有値）をまとめて求める。この1回の計算結果から、
        // 固有値が最大のものを第1主成分（Step3）、2番目に大きいものを第2主成分（Step4）として
        // それぞれ取り出す。ヤコビ法は任意サイズの対称行列に対応する。
        EigenResult eig = jacobiEigen(cov);

        // 固有値の大きい順に並べ替える。
        // 諸元数(m)にあわせて配列サイズを動的に決めるため、諸元が2個でも10個でも同じコードで対応できる。
        Integer[] order = new Integer[m];
        for (int i = 0; i < m; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> Double.compare(eig.values[y], eig.values[x]));

        double totalVar = 0;
        for (double v : eig.values) totalVar += v;

        // ---- Step3: 第1主成分の係数を求める ----
        // 固有値が最大の固有ベクトル＝データの分散を最も大きく説明する方向を第1主成分とする。
        int idx1 = order[0];
        double ratio1 = eig.values[idx1] / totalVar * 100;
        printComponentTable("Step3: 第1主成分の係数を求める", features, eig.values[idx1], ratio1, eig.vectors, idx1);

        // ---- Step4: 第2主成分の係数を求める ----
        // 固有値が2番目に大きい固有ベクトル＝第1主成分と直交する方向で、次に分散を説明する軸。
        int idx2 = order[1];
        double ratio2 = eig.values[idx2] / totalVar * 100;
        printComponentTable("Step4: 第2主成分の係数を求める", features, eig.values[idx2], ratio2, eig.vectors, idx2);

        // ---- Step5: 各データをPC1・PC2へ変換（＝次元削減の本体） ----
        // 固有値1位・2位に対応する固有ベクトルを軸として採用し、
        // 標準化済みデータをその2軸上へ内積によって投影する。
        // これにより m次元 → 2次元(PC1, PC2) への変換が完了する。
        double[][] pc = new double[m][2];
        for (int k = 0; k < 2; k++) {
            int idx = order[k];
            for (int a = 0; a < m; a++) pc[a][k] = eig.vectors[a][idx];
        }

        double[][] projected = new double[n][2];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 2; k++) {
                double s = 0;
                for (int a = 0; a < m; a++) s += z[i][a] * pc[a][k];
                projected[i][k] = s;
            }
        }

        printProjectedTable(names, projected);

        // ---- (参考) 寄与率 ----
        // 次元削減の本質的な処理ではなく、削減後にどれだけ元の情報を保持できているかを
        // 確認するための補足情報。
        System.out.println("## (参考) 寄与率\n");
        System.out.println("**累積寄与率（第1主成分＋第2主成分）: "
                + String.format("%.1f", ratio1 + ratio2) + "%**\n");

        // ---- FINAL: 結果の可視化（テキスト散布図） ----
        drawScatter(names, projected);
    }

    // ===================== 固有値分解（ヤコビ法） =====================

    /** 固有値分解の結果を保持する内部クラス。vectors は列ベクトルが固有ベクトル。 */
    static class EigenResult {
        double[] values;
        double[][] vectors;

        EigenResult(double[] v, double[][] vec) {
            values = v;
            vectors = vec;
        }
    }

    /**
     * 対称行列に対するヤコビ固有値法（任意サイズの n×n 対称行列に対応）。
     * 非対角成分のうち絶対値が大きい要素を選び、それが0になるような回転行列を
     * 繰り返し適用することで、最終的に対角行列（固有値）へ収束させる。
     */
    static EigenResult jacobiEigen(double[][] matrix) {
        int n = matrix.length;
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) A[i] = matrix[i].clone();

        double[][] V = new double[n][n];
        for (int i = 0; i < n; i++) V[i][i] = 1.0;

        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (i != j) off += A[i][j] * A[i][j];
            if (Math.sqrt(off) < 1e-10) break; // 非対角要素が十分小さくなったら収束とみなす

            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(A[p][q]) < 1e-12) continue;

                    double theta = 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
                    double c = Math.cos(theta), s = Math.sin(theta);

                    for (int k = 0; k < n; k++) {
                        double akp = A[k][p], akq = A[k][q];
                        A[k][p] = c * akp + s * akq;
                        A[k][q] = -s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++) {
                        double apk = A[p][k], aqk = A[q][k];
                        A[p][k] = c * apk + s * aqk;
                        A[q][k] = -s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = V[k][p], vkq = V[k][q];
                        V[k][p] = c * vkp + s * vkq;
                        V[k][q] = -s * vkp + c * vkq;
                    }
                }
            }
        }

        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) eigenvalues[i] = A[i][i];
        return new EigenResult(eigenvalues, V);
    }

    // ===================== 出力（Markdown表・テキスト散布図） =====================

    static void printTable(String title, String[] rowNames, String[] colNames, double[][] data) {
        System.out.println("## " + title + "\n");
        StringBuilder header = new StringBuilder("| 名前 |");
        StringBuilder sep = new StringBuilder("|---|");
        for (String c : colNames) {
            header.append(" ").append(c).append(" |");
            sep.append("---|");
        }
        System.out.println(header);
        System.out.println(sep);
        for (int i = 0; i < rowNames.length; i++) {
            StringBuilder row = new StringBuilder("| " + rowNames[i] + " |");
            for (int j = 0; j < data[i].length; j++) {
                row.append(String.format(" %.2f |", data[i][j]));
            }
            System.out.println(row);
        }
        System.out.println();
    }

    static void printMeanStd(String[] features, double[] mean, double[] std) {
        System.out.println("## Step1: 標準化パラメータ（各変数の平均・標準偏差）\n");
        System.out.println("| 変数 | 平均 | 標準偏差 |");
        System.out.println("|---|---|---|");
        for (int j = 0; j < features.length; j++) {
            System.out.println("| " + features[j] + " | "
                    + String.format("%.2f", mean[j]) + " | "
                    + String.format("%.2f", std[j]) + " |");
        }
        System.out.println();
    }

    static void printMatrix(String title, String[] labels, double[][] mat) {
        System.out.println("## " + title + "\n");
        StringBuilder header = new StringBuilder("| |");
        StringBuilder sep = new StringBuilder("|---|");
        for (String l : labels) {
            header.append(" ").append(l).append(" |");
            sep.append("---|");
        }
        System.out.println(header);
        System.out.println(sep);
        for (int i = 0; i < labels.length; i++) {
            StringBuilder row = new StringBuilder("| " + labels[i] + " |");
            for (int j = 0; j < labels.length; j++) row.append(String.format(" %.3f |", mat[i][j]));
            System.out.println(row);
        }
        System.out.println();
    }

    /**
     * 1つの主成分（第1主成分 または 第2主成分）の係数（固有ベクトルの成分）を
     * 諸元ごとに表示する。Step3・Step4でそれぞれ1回ずつ呼び出す。
     */
    static void printComponentTable(String title, String[] features, double eigenvalue,
                                     double ratio, double[][] eigenvectors, int idx) {
        System.out.println("## " + title + "\n");
        System.out.println("固有値: " + String.format("%.3f", eigenvalue)
                + " ／ 寄与率: " + String.format("%.1f", ratio) + "%\n");
        System.out.println("| 諸元 | 係数（固有ベクトル成分） |");
        System.out.println("|---|---|");
        for (int a = 0; a < features.length; a++) {
            System.out.println("| " + features[a] + " | "
                    + String.format("%.3f", eigenvectors[a][idx]) + " |");
        }
        System.out.println();
    }

    static void printProjectedTable(String[] names, double[][] projected) {
        System.out.println("## Step5: 各データをPC1・PC2へ変換（主成分得点）\n");
        System.out.println("| 名前 | PC1 | PC2 |");
        System.out.println("|---|---|---|");
        for (int i = 0; i < names.length; i++) {
            System.out.println("| " + names[i] + " | "
                    + String.format("%.3f", projected[i][0]) + " | "
                    + String.format("%.3f", projected[i][1]) + " |");
        }
        System.out.println();
    }

    static void drawScatter(String[] names, double[][] p) {
        int width = 50, height = 20;
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE, maxY = -Double.MAX_VALUE;
        for (double[] pt : p) {
            minX = Math.min(minX, pt[0]);
            maxX = Math.max(maxX, pt[0]);
            minY = Math.min(minY, pt[1]);
            maxY = Math.max(maxY, pt[1]);
        }
        double mx = (maxX - minX) * 0.15 + 0.01;
        double my = (maxY - minY) * 0.15 + 0.01;
        minX -= mx; maxX += mx; minY -= my; maxY += my;

        char[][] grid = new char[height][width];
        for (char[] row : grid) Arrays.fill(row, ' ');

        // データ件数(n)がいくつになっても対応できるよう記号を動的に生成する。
        // 26件まではアルファベット1文字（A,B,C…Z）で一意に識別できるが、
        // それを超える場合はテキスト格子1マスに複数文字は収まらないため、
        // グラフ上は共通の記号(*)で位置のみ示し、正確な値は下の凡例表（連番）で確認する。
        String[] symbols = new String[p.length];
        for (int i = 0; i < p.length; i++) {
            symbols[i] = (p.length <= 26) ? String.valueOf((char) ('A' + i)) : "*";
        }

        for (int i = 0; i < p.length; i++) {
            int cx = (int) Math.round((p[i][0] - minX) / (maxX - minX) * (width - 1));
            int cy = (int) Math.round((p[i][1] - minY) / (maxY - minY) * (height - 1));
            int row = height - 1 - cy; // 上に行くほどPC2が大きくなるように反転
            grid[row][cx] = symbols[i].charAt(0);
        }

        System.out.println("## FINAL: グラフ（PC1 - PC2 散布図）\n");
        System.out.println("```");
        System.out.println("PC2");
        for (int r = 0; r < height; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < width; c++) sb.append(grid[r][c]);
            System.out.println(sb);
        }
        StringBuilder axis = new StringBuilder();
        for (int c = 0; c < width; c++) axis.append("-");
        System.out.println(axis + "> PC1");
        System.out.println("```\n");

        System.out.println("**凡例**（27件以上の場合、グラフ上の記号は位置確認用の共通マーク(*)になるため、"
                + "この表の連番と名前で個々のデータを識別してください）\n");
        System.out.println("| No. | 記号 | 名前 | PC1 | PC2 |");
        System.out.println("|---|---|---|---|---|");
        for (int i = 0; i < p.length; i++) {
            System.out.println("| " + (i + 1) + " | " + symbols[i] + " | " + names[i] + " | "
                    + String.format("%.3f", p[i][0]) + " | "
                    + String.format("%.3f", p[i][1]) + " |");
        }
    }
}
