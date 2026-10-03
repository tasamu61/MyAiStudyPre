import java.util.*;

/**
 * 16x16(256次元)ドットパターンの次元削減実験を、Colabで作ったPythonノートブックから
 * Javaに移植したもの。1ファイルに、必要な部品を内部クラスとしてまとめている。
 *
 * 構成(Pythonノートブックの節に対応):
 *   Step1: 基本パターン(十字・丸・斜め線・市松模様・バツ・ランダム)の生成
 *   Step2: ノイズを混ぜた見た目の確認(ASCIIプレビュー)
 *   Step3: 学習データ・推論用データの作成
 *   Step4: PCAで256次元->K次元に圧縮(デュアルトリック: n×nのグラム行列を使う)
 *   Step5: 256次元のまま判定する分類器 と K次元に減らして判定する分類器の比較
 *   Step6: 量子化(K次元の値を何ビットで表すか)
 *   Step7: (参考)符号付きランダム射影(最初からビットを出す方法)
 *
 * 注意: このファイルはJDK(javac)が使えない環境で作成したため、実際のコンパイルは
 * 行えていない。ロジックは、同じ内容をPythonで検証した結果と対応するように書いている。
 */
public class DimReductionDemo {

    static final int N = 16;      // 16x16
    static final int D = N * N;   // 256次元

    // ===================== Step1: 基本パターン =====================

    static double[] patternCross() {
        double[] a = new double[D];
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                a[y * N + x] = (x == 7 || x == 8 || y == 7 || y == 8) ? 1.0 : 0.0;
        return a;
    }

    static double[] patternCircle() {
        double[] a = new double[D];
        double cx = (N - 1) / 2.0, cy = (N - 1) / 2.0;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                double dx = x - cx, dy = y - cy;
                a[y * N + x] = (Math.sqrt(dx * dx + dy * dy) <= 6) ? 1.0 : 0.0;
            }
        return a;
    }

    static double[] patternDiag() {
        double[] a = new double[D];
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                a[y * N + x] = (Math.abs((N - 1 - y) - x) <= 1) ? 1.0 : 0.0;
        return a;
    }

    static double[] patternChecker() {
        double[] a = new double[D];
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                a[y * N + x] = (((x / 2) + (y / 2)) % 2 == 0) ? 1.0 : 0.0;
        return a;
    }

    static double[] patternX() {
        double[] a = new double[D];
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                boolean v = Math.abs((N - 1 - y) - x) <= 1 || Math.abs(y - x) <= 1;
                a[y * N + x] = v ? 1.0 : 0.0;
            }
        return a;
    }

    /** 1件分のデータ: ラベル(クラス名)と256次元の画素値。 */
    record Data(String label, double[] values) {
    }

    /** 基本パターンにノイズ(確率flipProbで画素を反転)を混ぜた1件を作る。 */
    static double[] makeSample(String cls, Map<String, double[]> base, double flipProb, Random rng) {
        double[] img;
        if (cls.equals("ランダム")) {
            img = new double[D];
            for (int i = 0; i < D; i++) img[i] = rng.nextDouble() > 0.5 ? 1.0 : 0.0;
        } else {
            img = base.get(cls).clone();
        }
        double[] out = new double[D];
        for (int i = 0; i < D; i++) {
            boolean flip = rng.nextDouble() < flipProb;
            out[i] = flip ? 1.0 - img[i] : img[i];
        }
        return out;
    }

    static List<Data> makeDataset(String[] classes, Map<String, double[]> base,
                                   int nPerClass, double flipProb, Random rng) {
        List<Data> list = new ArrayList<>();
        for (String c : classes)
            for (int i = 0; i < nPerClass; i++)
                list.add(new Data(c, makeSample(c, base, flipProb, rng)));
        return list;
    }

    static void printAscii(String label, double[] img) {
        System.out.println(label + ":");
        for (int y = 0; y < N; y++) {
            StringBuilder sb = new StringBuilder();
            for (int x = 0; x < N; x++) sb.append(img[y * N + x] >= 0.5 ? "■" : "・");
            System.out.println(sb);
        }
    }

    // ===================== 固有値分解(ヤコビ法。既出のPCA2D.javaと同じ実装) =====================

    static class EigenResult {
        double[] values;
        double[][] vectors;
        EigenResult(double[] v, double[][] vec) { values = v; vectors = vec; }
    }

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
            if (Math.sqrt(off) < 1e-8) break;

            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(A[p][q]) < 1e-12) continue;
                    double theta = 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
                    double c = Math.cos(theta), s = Math.sin(theta);
                    for (int k = 0; k < n; k++) {
                        double akp = A[k][p], akq = A[k][q];
                        A[k][p] = c * akp + s * akq; A[k][q] = -s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++) {
                        double apk = A[p][k], aqk = A[q][k];
                        A[p][k] = c * apk + s * aqk; A[q][k] = -s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = V[k][p], vkq = V[k][q];
                        V[k][p] = c * vkp + s * vkq; V[k][q] = -s * vkp + c * vkq;
                    }
                }
            }
        }
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) eigenvalues[i] = A[i][i];
        return new EigenResult(eigenvalues, V);
    }

    // ===================== Step4: PCA(デュアルトリック) =====================
    // 諸元数D(256)がデータ件数nよりずっと多いため、D×Dの共分散行列ではなく、
    // n×nのグラム行列を固有値分解する「デュアルトリック」を使う(HTMLデモと同じ考え方)。
    static class PcaDual {
        int K, n;
        double[] mean;
        double[][] Xc;      // 中心化した学習データ(n×D)。推論データの変換に使う
        double[] vals;      // 上位K個の固有値
        double[][] vecs;    // 対応する固有ベクトル(n×K、列がベクトル)
        double totalVar;    // 全固有値の合計(寄与率の分母)

        void fit(double[][] X, int K) {
            this.K = K; this.n = X.length; int Ddim = X[0].length;
            mean = new double[Ddim];
            for (double[] row : X) for (int j = 0; j < Ddim; j++) mean[j] += row[j] / n;
            Xc = new double[n][Ddim];
            for (int i = 0; i < n; i++) for (int j = 0; j < Ddim; j++) Xc[i][j] = X[i][j] - mean[j];

            double[][] G = new double[n][n];
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) {
                    double s = 0; for (int k = 0; k < Ddim; k++) s += Xc[i][k] * Xc[j][k];
                    G[i][j] = G[j][i] = s / n;
                }
            EigenResult eig = jacobiEigen(G);

            totalVar = 0; for (double v : eig.values) totalVar += Math.max(v, 0);

            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(eig.values[b], eig.values[a]));

            vals = new double[K];
            vecs = new double[n][K];
            for (int k = 0; k < K; k++) {
                int idx = order[k];
                vals[k] = eig.values[idx];
                for (int i = 0; i < n; i++) vecs[i][k] = eig.vectors[i][idx];
            }
        }

        /** 学習データ自身をK次元に変換した値(= sqrt(n*λ) * 固有ベクトル、デュアルトリックの公式)。 */
        double[][] transformTrain() {
            double[][] out = new double[n][K];
            for (int i = 0; i < n; i++)
                for (int k = 0; k < K; k++)
                    out[i][k] = Math.sqrt(Math.max(n * vals[k], 0)) * vecs[i][k];
            return out;
        }

        /** 新しいデータ(推論用など)をK次元に変換する。学習データとの内積を経由する。 */
        double[][] transform(double[][] X) {
            int m = X.length, Ddim = mean.length;
            double[][] out = new double[m][K];
            for (int t = 0; t < m; t++) {
                double[] xc = new double[Ddim];
                for (int d = 0; d < Ddim; d++) xc[d] = X[t][d] - mean[d];
                double[] dots = new double[n];
                for (int i = 0; i < n; i++) {
                    double s = 0; for (int d = 0; d < Ddim; d++) s += Xc[i][d] * xc[d];
                    dots[i] = s;
                }
                for (int k = 0; k < K; k++) {
                    double lam = vals[k];
                    if (lam <= 1e-9) { out[t][k] = 0; continue; }
                    double s = 0; for (int i = 0; i < n; i++) s += dots[i] * vecs[i][k];
                    out[t][k] = s / Math.sqrt(n * lam);
                }
            }
            return out;
        }
    }

    // ===================== k近傍法(k-NN) =====================

    static class Knn {
        double[][] X; String[] y;
        void fit(double[][] X, String[] y) { this.X = X; this.y = y; }

        String predict(double[] x, int k) {
            int n = X.length;
            Integer[] idx = new Integer[n];
            double[] d = new double[n];
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < x.length; j++) { double diff = X[i][j] - x[j]; s += diff * diff; }
                d[i] = s; idx[i] = i;
            }
            Arrays.sort(idx, (a, b) -> Double.compare(d[a], d[b]));
            Map<String, Integer> cnt = new HashMap<>();
            for (int t = 0; t < k; t++) cnt.merge(y[idx[t]], 1, Integer::sum);
            String best = null; int bc = -1;
            for (Map.Entry<String, Integer> e : cnt.entrySet())
                if (e.getValue() > bc) { bc = e.getValue(); best = e.getKey(); }
            return best;
        }
    }

    // ===================== Step6: 量子化 =====================

    static double[] colMin(double[][] X) {
        double[] m = X[0].clone();
        for (double[] row : X) for (int j = 0; j < row.length; j++) m[j] = Math.min(m[j], row[j]);
        return m;
    }
    static double[] colMax(double[][] X) {
        double[] m = X[0].clone();
        for (double[] row : X) for (int j = 0; j < row.length; j++) m[j] = Math.max(m[j], row[j]);
        return m;
    }
    /** 連続値を2^bits段階の整数に丸める(量子化)。 */
    static double[][] quantize(double[][] X, int bits, double[] vmin, double[] vmax) {
        double levels = Math.pow(2, bits) - 1;
        double[][] q = new double[X.length][X[0].length];
        for (int i = 0; i < X.length; i++)
            for (int j = 0; j < X[0].length; j++) {
                double v = (X[i][j] - vmin[j]) / (vmax[j] - vmin[j] + 1e-12) * levels;
                v = Math.rint(v);
                q[i][j] = Math.max(0, Math.min(levels, v));
            }
        return q;
    }
    /** 量子化した整数を、判定に使うためもとの範囲に戻す。 */
    static double[][] dequantize(double[][] Q, int bits, double[] vmin, double[] vmax) {
        double levels = Math.pow(2, bits) - 1;
        double[][] out = new double[Q.length][Q[0].length];
        for (int i = 0; i < Q.length; i++)
            for (int j = 0; j < Q[0].length; j++)
                out[i][j] = Q[i][j] / levels * (vmax[j] - vmin[j]) + vmin[j];
        return out;
    }

    // ===================== Step7: 符号付きランダム射影 =====================

    static class SignedRandomProjection {
        double[][] planes; // D x B
        void fit(int Ddim, int B, Random rng) {
            planes = new double[Ddim][B];
            for (int i = 0; i < Ddim; i++) for (int j = 0; j < B; j++) planes[i][j] = rng.nextGaussian();
        }
        /** 内積の符号だけを0.0/1.0のビットとして返す(学習データを一切見ない)。 */
        double[][] transform(double[][] X) {
            int m = X.length, Ddim = X[0].length, B = planes[0].length;
            double[][] out = new double[m][B];
            for (int t = 0; t < m; t++)
                for (int b = 0; b < B; b++) {
                    double s = 0; for (int d = 0; d < Ddim; d++) s += X[t][d] * planes[d][b];
                    out[t][b] = s > 0 ? 1.0 : 0.0;
                }
            return out;
        }
    }

    // ===================== 補助関数 =====================

    static double[][] toMatrix(List<Data> list) {
        double[][] m = new double[list.size()][];
        for (int i = 0; i < list.size(); i++) m[i] = list.get(i).values();
        return m;
    }
    static String[] toLabels(List<Data> list) {
        String[] y = new String[list.size()];
        for (int i = 0; i < list.size(); i++) y[i] = list.get(i).label();
        return y;
    }
    static double accuracy(String[] pred, String[] trueY) {
        int c = 0;
        for (int i = 0; i < pred.length; i++) if (pred[i].equals(trueY[i])) c++;
        return 100.0 * c / pred.length;
    }

    // ===================== main =====================

    public static void main(String[] args) {
        Random rng = new Random(42); // 実験データ用の乱数(ノイズ体感とは分けていないので再実行すると値は変わる点に注意)
        String[] classes = {"十字", "丸", "斜め線", "市松模様", "バツ", "ランダム"};
        Map<String, double[]> base = new LinkedHashMap<>();
        base.put("十字", patternCross());
        base.put("丸", patternCircle());
        base.put("斜め線", patternDiag());
        base.put("市松模様", patternChecker());
        base.put("バツ", patternX());
        // 「ランダム」は固定テンプレートを持たない(makeSampleの中で毎回新規生成する)

        System.out.println("=== Step1: 基本パターン ===");
        for (String c : classes) {
            double[] img = c.equals("ランダム") ? makeSample(c, base, 0.0, rng) : base.get(c);
            printAscii(c, img);
        }

        System.out.println("\n=== Step2: ノイズを混ぜた見た目(0%, 10%, 20%) ===");
        for (double p : new double[]{0.0, 0.10, 0.20}) {
            System.out.println("-- ノイズ" + (int) Math.round(p * 100) + "% --");
            for (String c : classes) printAscii(c, makeSample(c, base, p, rng));
        }

        // ---- 実験パラメータ(ここをまとめて変更できる) ----
        int K = 16;
        int nTrainPerClass = 20; double trainNoise = 0.05;
        int nTestPerClass = 20; double testNoise = 0.10;

        List<Data> trainList = makeDataset(classes, base, nTrainPerClass, trainNoise, rng);
        List<Data> testList = makeDataset(classes, base, nTestPerClass, testNoise, rng);
        double[][] Xtrain = toMatrix(trainList), Xtest = toMatrix(testList);
        String[] ytrain = toLabels(trainList), ytest = toLabels(testList);

        System.out.println("\n=== Step3: データセット ===");
        System.out.printf("学習: %d件 x %d次元 (ノイズ%.0f%%)%n", Xtrain.length, Xtrain[0].length, trainNoise * 100);
        System.out.printf("推論: %d件 x %d次元 (ノイズ%.0f%%、学習には未使用)%n", Xtest.length, Xtest[0].length, testNoise * 100);

        System.out.println("\n=== Step4: PCAで256->" + K + "次元 ===");
        PcaDual pca = new PcaDual();
        pca.fit(Xtrain, K);
        double[][] XtrainK = pca.transformTrain();
        double[][] XtestK = pca.transform(Xtest);
        double sumK = 0; for (double v : pca.vals) sumK += Math.max(v, 0);
        System.out.printf("累積寄与率: %.1f%%%n", sumK / pca.totalVar * 100);

        Knn knn256 = new Knn(); knn256.fit(Xtrain, ytrain);
        Knn knnK = new Knn(); knnK.fit(XtrainK, ytrain);
        String[] pred256 = new String[Xtest.length], predK = new String[Xtest.length];
        for (int i = 0; i < Xtest.length; i++) {
            pred256[i] = knn256.predict(Xtest[i], 3);
            predK[i] = knnK.predict(XtestK[i], 3);
        }

        System.out.println("\n=== Step5: クラスごとの正解率(256次元 vs " + K + "次元) ===");
        for (String c : classes) {
            int n = 0, c256 = 0, cK = 0, agree = 0;
            for (int i = 0; i < ytest.length; i++) {
                if (!ytest[i].equals(c)) continue;
                n++;
                if (pred256[i].equals(c)) c256++;
                if (predK[i].equals(c)) cK++;
                if (pred256[i].equals(predK[i])) agree++;
            }
            System.out.printf("%-6s %3d件  256次元=%6.1f%%  %d次元=%6.1f%%  一致=%6.1f%%%n",
                    c, n, 100.0 * c256 / n, K, 100.0 * cK / n, 100.0 * agree / n);
        }

        System.out.println("\n=== Step6: 量子化 ===");
        double acc256 = accuracy(pred256, ytest);
        System.out.printf("%-26s %8s %8s%n", "表現", "1件(B)", "正解率");
        System.out.printf("%-26s %7.1fB %7.1f%%%n", "256次元(削減前,1bit/画素)", D / 8.0, acc256);
        double[] vmin = colMin(XtrainK), vmax = colMax(XtrainK);
        for (int bits : new int[]{64, 8, 6, 4, 3, 2, 1}) {
            double[][] trQ, teQ; double nbytes;
            if (bits == 64) { trQ = XtrainK; teQ = XtestK; nbytes = K * 8; }
            else {
                trQ = dequantize(quantize(XtrainK, bits, vmin, vmax), bits, vmin, vmax);
                teQ = dequantize(quantize(XtestK, bits, vmin, vmax), bits, vmin, vmax);
                nbytes = K * bits / 8.0;
            }
            Knn knnQ = new Knn(); knnQ.fit(trQ, ytrain);
            String[] predQ = new String[Xtest.length];
            for (int i = 0; i < Xtest.length; i++) predQ[i] = knnQ.predict(teQ[i], 3);
            double accQ = accuracy(predQ, ytest);
            String label = bits == 64 ? K + "次元 float64(量子化なし)" : K + "次元 " + bits + "bit量子化";
            System.out.printf("%-26s %7.1fB %7.1f%%%n", label, nbytes, accQ);
        }

        System.out.println("\n=== Step7: (参考)符号付きランダム射影 ===");
        Random srpRng = new Random(7);
        for (int B : new int[]{16, 8, 4, 2}) {
            SignedRandomProjection srp = new SignedRandomProjection();
            srp.fit(D, B, srpRng);
            double[][] Btr = srp.transform(Xtrain), Bte = srp.transform(Xtest);
            Knn knnB = new Knn(); knnB.fit(Btr, ytrain);
            String[] predB = new String[Xtest.length];
            for (int i = 0; i < Xtest.length; i++) predB[i] = knnB.predict(Bte[i], 3);
            double accB = accuracy(predB, ytest);
            System.out.printf("%2dbit (%.1fB): 正解率=%.1f%%%n", B, B / 8.0, accB);
        }
        System.out.printf("(比較) 256次元そのまま: %.1f%%%n", acc256);
    }
}
