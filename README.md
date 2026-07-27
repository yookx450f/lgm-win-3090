# LGM Win 3090 - 3D Car Model Generator

複数の画像から車の3Dモデリング（.glb, .obj）を生成するシステムです。
WindowsのWSL2上でDocker経由動作し、RTX3090 GPUを使用して高性能な3D復元を行います。

## 概要

このシステムは、以下のパイプラインを使用して車の3Dモデルを生成します：

```
[複数画像]
      ↓
[COLMAP：カメラ推定 & 点群生成]
      ↓
[Gaussian Splatting：高品質3D復元]
      ↓
[Mesh化：Poisson / Instant Meshes]
      ↓
[glb / obj 出力]
      ↓
[Blenderで車比較動画を作成]
```

## 処理詳細

### 1. 前処理（Preprocessing Layer）
- 画像の正規化
- 車体のマスク生成（背景除去）
- 画像の整列

### 2. COLMAP：カメラ推定 & 点群生成
- Structure-from-Motion (SfM)
- カメラパラメータの推定
- 疎点群（Sparse Point Cloud）生成
- 画像間の対応点を検出

### 3. Dense Reconstruction（密点群生成）
- 密点群（Dense Point Cloud）生成
- 多視点ステレオ（Multi-View Stereo）
- 詳細な形状の復元

### 4. Gaussian Splatting：高品質3D復元
- 3Dガウス分布の最適化
- 高品質なレンダリング
- テクスチャの高精度な表現
- 反射・光沢の再現

### 5. Mesh化（Meshing）
- Poisson Surface Reconstruction
- Instant Meshes
- 点群 → メッシュ（Mesh）化
- メッシュの平滑化

### 6. Texture Baking（テクスチャ生成）
- UV展開
- テクスチャマッピング
- 色補正
- 光沢（Specular）
- 反射（Reflection）

### 7. 出力（Export Layer）
最終的に以下の形式で出力：
- glb（YouTube向けに最適）
- obj
- ply

### 8. Blenderで車比較動画を作成
- 複数の3Dモデルを比較
- YouTube向けの動画を作成
- アニメーションの追加

## 要件

- **OS**: Windows 10/11 with WSL2 (Ubuntu 22.04推奨)
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **Software**:
  - Docker Desktop for Windows
  - NVIDIA Container Toolkit
  - Git

## ディレクトリ構造

```
lgm-win-3090/
├── Dockerfile              # Dockerイメージの定義
├── docker-compose.yml      # Docker Compose設定（GPU対応）
├── .dockerignore           # Docker除外ファイル
├── build.sh               # ビルドスクリプト
├── run_car_model.sh       # 実行スクリプト
├── .clinerules            # プロジェクトルール
├── input/                 # 入力画像ディレクトリ
├── output/                # 出力3Dモデルディレクトリ
├── cache/                 # キャッシュディレクトリ
├── workspace/             # 作業ディレクトリ
└── scripts/               # パイプラインスクリプト
    ├── preprocess.py      # 前処理スクリプト
    ├── meshing.py         # メッシュ化スクリプト
    └── texture_baking.py  # テクスチャベイクスクリプト
```

## セットアップ

### 1. 環境確認

```bash
# 環境を確認
./build.sh check
```

### 2. Dockerイメージのビルド

```bash
# イメージをビルド
./build.sh build
```

> **注意**: 初回ビルドには時間がかかります（PyTorch、COLMAP、依存パッケージのダウンロードのため）。

### 3. 3Dモデルの生成

```bash
# 既定のディレクトリを使用
./run_car_model.sh

# カスタムディレクトリを指定
./run_car_model.sh /path/to/input /path/to/output
```

## 使用法

### ビルドスクリプトのオプション

```bash
./build.sh [COMMAND]

コマンド:
  build    Dockerイメージをビルド
  run      3Dモデル生成を実行
  shell    インタラクティブシェルを開始
  stop     コンテナを停止
  clean    コンテナとイメージを削除
  check    環境要件を確認
  help     ヘルプメッセージを表示
```

### 実行スクリプトのオプション

```bash
./run_car_model.sh [INPUT_DIR] [OUTPUT_DIR]

引数:
  INPUT_DIR    車画像のディレクトリ（既定: ./input）
  OUTPUT_DIR   出力3Dモデルのディレクトリ（既定: ./output）
```

### 画像の準備

1. `input/` ディレクトリに車の写真を配置
2. 形式: `.jpg`, `.jpeg`, `.png`
3. 複数の角度からの写真を推奨（正面、背面、側面、斜めなど）

## コンテナ構造

### メインサービス（car-model）

- 3Dモデル生成パイプラインを実行
- 入力画像から.glbファイルを生成

### シェルサービス（car-model-shell）

- デバッグ用のインタラクティブシェル
- コンテナ内で手動コマンドを実行可能

## 出力形式

- **glb**: YouTube向けに最適化された3Dフォーマット
- **obj**: オープンな3Dフォーマット
- **ply**: ポイントクラウドフォーマット

## パイプラインスクリプト

### [`scripts/preprocess.py`](scripts/preprocess.py:1)
画像の前処理：
- 画像の正規化
- 車体のマスク生成（背景除去）
- 画像の整列

### [`scripts/meshing.py`](scripts/meshing.py:1)
メッシュ化：
- Poisson Surface Reconstruction
- Instant Meshes
- 点群 → メッシュ（Mesh）化

### [`scripts/texture_baking.py`](scripts/texture_baking.py:1)
テクスチャベイク：
- UV展開
- テクスチャマッピング
- 色補正
- 光沢・反射の処理

## トラブルシューティング

### GPUが認識されない

```bash
# NVIDIA Container Toolkitがインストールされているか確認
docker info | grep -i nvidia

# GPUが認識されているか確認
nvidia-smi
```

### ビルドが失敗する

```bash
# キャッシュをクリアして再ビルド
./build.sh clean
./build.sh build
```

### メモリ不足

RTX3090の24GB VRAMを使用するため、以下の設定を確認:
- Docker Desktopのメモリ割り当てを8GB以上に設定
- WSL2のメモリ制限を確認（`.wslconfig`）

## ファイル構成

| ファイル | 説明 |
|---------|------|
| [`Dockerfile`](Dockerfile) | Dockerイメージの定義（CUDA 12.x対応） |
| [`docker-compose.yml`](docker-compose.yml) | Docker Compose設定（GPU対応） |
| [`.dockerignore`](.dockerignore) | Docker除外ファイル |
| [`build.sh`](build.sh) | ビルドスクリプト |
| [`run_car_model.sh`](run_car_model.sh) | 実行スクリプト |
| [`.clinerules`](.clinerules) | プロジェクトルール |
| [`scripts/preprocess.py`](scripts/preprocess.py) | 前処理スクリプト |
| [`scripts/meshing.py`](scripts/meshing.py) | メッシュ化スクリプト |
| [`scripts/texture_baking.py`](scripts/texture_baking.py) | テクスチャベイクスクリプト |

## ライセンス

このプロジェクトはMITライセンスの下で提供されます。

## 作者

LGM Win 3090 Project
