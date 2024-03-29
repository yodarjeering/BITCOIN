# 仕様書


## BitCoinの自動取引における種々のコード
本研究では, coincheck の API および, マーケットメイク戦略を用いた自動売買を行った. 自動売買に必要なコードなどを記載してある.
<br>

## 動作環境

OS : Windows 11以上
<br>

## 準備
以下のツールを必要とする.


<li> Python3
<li> matplotlib
<li> numpy
<li> seaborn
<li> nath
<li> pandas 
<li> json
<li> request
<li> glob
<li> sys
<li> ccxt
<li> datetime
<li> pprint
<li> logging
<li> time
<br>
<br>
基本的に以下のようにインストールする
<br>
<br>

```bash
pip install huga_package
```

Python3は Anaconda 環境をインストールして, pip コマンドを使えるようにする.
## ディレクトリ構成

```bash
BitCoin
├─Code
│  ├─my_library
│  │ 
│  └─coincheck.ipynb   
├─Data
└─Log
```
## 各ディレクトリ, ファイル説明

* coincheck.ipynb : 実際に自動売買システムを動かすファイル
* Data : 検証用データのあるディレクトリ
* Log : hoge.log ファイルを保存するディレクトリ

## 使用法

主にcoincheck.ipynbに使用法を記載している. そちらを参照されたい. 

## Author

作成情報

* github : https://github.com/yodarjeering/BITCOIN

## 参考文献

  1.   執行戦略と取引コストに関する研究の進展 <br> 売買戦略について詳細にまとめてある. 方針が経たないうちは熟読すべし. 筆者はかなり参考にした. 本研究では, マーケットメイク戦略を用いた.
   https://www.imes.boj.or.jp/research/papers/japanese/kk31-1-8.pdf

## BitCoin研究の所感
* BitCoinで儲けるのはきつい.<br>
 筆者は2022/09/21-2023/02/16までシステムを動かしてみたが, FTX 事件 (2022/11/08-) の大暴落で, およそ２か月分の利益が吹っ飛んだ. その後もソースコードの改善など続けてシステムを動かすも, 損失がかさんだ. 2023/02/16に BitCoin 撤退を表明. 以降研究はしていない.
* BitCoin は古典的金融理論が通用しない.<br>
  株や, 為替は古くから研究がなされ, それらの理論はある程度, 株や為替には通用すると考える. しかし, BitCoinをはじめとする仮想通貨は, 株や為替と異なり, 法的な裏付けが存在しないため, 現代までの金融理論が通用しない可能性がある.　ボラティリティも株や為替では考えられないほど大きい. したがって, 仮想通貨は投機として側面が強く, およそ投資とは言えない.
* 上昇相場ではどんなアルゴリズムでも儲かる<br>
これは株でも為替でも同じことが言えるが, 上昇相場ならどんな悪いアルゴリズムでも一定の収益はえられるというのが筆者の感想. 下降相場は推して知るべし. 
<br>
<br>

これらはあくまでも個人的な感想. BitCoin と相関の弱いアルトコインなどではもしかしたらうまくいくかもしれない. また, マーケットメイク戦略は, 高頻度取引戦略だったがそれ以外の戦略で試してみるともしかしたらうまくいくかもしれない. また, 仮想通貨相場がこの先も下降基調であると仮定すれば, 売り注文から入って, それよりも安値で買いなおせばそのマージンでうまく稼げるかもしれない.

