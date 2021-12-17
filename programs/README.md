heatmap.py 楽曲をヒートマップとして可視化
heatmap.pyを使用する際にffmpeg,pydub,pandas,seaborn,matplotlibのインストールが必要
ffmpegインストール方法
https://jp.videoproc.com/edit-convert/how-to-download-and-install-ffmpeg.html
  必要パッケージのインストールは  
pip3 install pydub  
pip3 install pandas  
pip3 install seaborn  
pip3 install matplotlib


categorize.py　ジャンル推定モデルの作成  
categorize.pyを利用する際にlibrosaのインストール  
$ sudo apt install llvm-7  
$LLVM_CONFIG=/usr/bin/llvm-config-7   
$pip3 install llvmlite==0.31.0   
$pip3 install numba==0.48.0   
$pip3 install colorama==0.3.9   
$pip3 install librosa==0.6.3  
$pip3 install keras == 2.3.1  
$pip3 install tensorflow == 2.3.1  
$pip3 install h5py == 2.10.0

pip3 でインストールした場合Python3で実行する必要がある.
  
モデル作成のデータセットには下記のfma_smallを使用  
https://github.com/mdeff/fma#data

genre.py  
heatmap.pyとcategorize.pyを合わせて改良したもの。実際に楽曲のジャンル推定を行うプログラム．  
実際の動きとしては  
1.楽曲を画像として可視化  
2.画像から特徴を抽出してcsvとして保存  
3.抽出した特徴から楽曲ジャンル推定モデルの作成  
4.楽曲ジャンル推定モデルから楽曲ジャンルの推定  
必要なモジュールは上記のもがあれば動作する。
