public class Main06_2 extends Main06_1  {



    // 諸元名
    String[] featureNames = {
        "排気量(cc)",
        "馬力(ps)",
        "トルク",
        "重量(kg)",
        "価格(万円)",
        "燃費(km/L)",
        "乗員数(人)"
    };

    // 名前 ＋ 諸元
    Data[] data = {
        new Data("軽自動車",     new double[]{ 660,  40, 9,   880, 150, 20, 4}),
        new Data("コンパクト",   new double[]{1000,  80, 11, 1050, 200, 26, 5}),
        new Data("ヴァン",       new double[]{2000, 150, 23, 1500, 230, 11, 7}),
        new Data("セダン",       new double[]{2000, 140, 24, 1200, 240, 16, 5}),
        new Data("RV",          new double[]{2800, 150, 33, 1550, 300, 12, 5}),
        new Data("オープンカー",  new double[]{2000, 140, 22, 1000, 250, 18, 2}),
        new Data("スポーツカー",  new double[]{2500, 180, 30, 1100, 300, 14, 2}),
        new Data("高級車",       new double[]{3000,  180, 33, 1400, 400, 12, 5})
    };

    public static void main(String[] args) {


        Main06_2 main = new Main06_2();
        main.doMain(main.featureNames, main.data);
    }   

}