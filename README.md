# Python_Act_Dobot# Conda 仮想環境の構築

```pmd
$ conda create -n dobot python=3.8
```
で python version 3.8 の仮想環境を構築する．
次に，
```pmd
$ conda activate dobot
```
を実行して，仮想環境をアクティブに設定する．

# 仮想環境のアップデート

以下のサイトを参考にした．
[Pythonの仮想環境管理（conda使用）とパッケージ管理（pip）のまとめ](https://arakan-pgm-ai.hatenablog.com/entry/2019/07/01/000000)

## conda 自体のアップデート

```cmd
conda update -n base conda
```

## conda に含まれるパッケージをすべてアップデート

```cmd
conda update --all
```

## 仮想環境の python のアップデート

```cmd
conda update python
```

## 仮想環境の pip 自身のアップデート

```cmd
pip3 install -U pip3
```

## 仮想環境の pip パッケージのアップデート

```cmd
pip3 install -U パッケージ名
```

## 仮想環境にインストールした pip パッケージの書き出し

```cmd
pip3 freeze > requirements.txt
```

## 書き出したパッケージを仮想環境にインストール

```cmd
pip3 install -r requirements.txt
```