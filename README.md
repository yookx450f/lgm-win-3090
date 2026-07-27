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
    ├── colmap.py          # COLMAP処理スクリプト
    ├── gaussian_splatting.py  # Gaussian Splattingスクリプト
    ├── meshing.py         # メッシュ化スクリプト
    ├── texture_baking.py  # テクスチャベイクスクリプト
    ├── blender_video.py   # Blender動画生成スクリプト
    └── run_pipeline.py    # メインパイプラインスクリプト
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

## Web API

### アクセス方法

FastAPIサーバーを起動すると、以下のURLでアクセスできます：

- **管理画面**: http://localhost:8000
- **APIドキュメント**: http://localhost:8000/docs
- **JSON API**: http://localhost:8000/api/

### 主要エンドポイント

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/` | GET | 管理画面 |
| `/api/jobs` | GET | ジョブ一覧を取得 |
| `/api/jobs/{job_id}` | GET | 特定のジョブ状態を取得 |
| `/api/upload` | POST | 画像をアップロード |
| `/api/pipeline/{job_id}/start` | POST | パイプラインを実行 |
| `/api/jobs/{job_id}/cancel` | POST | ジョブをキャンセル |
| `/api/results/{job_id}` | GET | 結果ファイル一覧を取得 |
| `/api/download/{job_id}/{filename}` | GET | ファイルをダウンロード |
| `/api/viewer/{job_id}` | GET | 3Dモデルビューアを表示 |

### 使用法

#### Web APIを使用する場合

1. **FastAPIサーバーを起動**
   ```bash
   docker compose up car-api
   ```

2. **ブラウザでアクセス**
   - http://localhost:8000 で管理画面を開く

3. **画像をアップロード**
   - 管理画面から車の写真を複数選択してアップロード

4. **パイプラインを実行**
   - 設定を選択して「パイプライン実行」ボタンをクリック

5. **結果を確認**
   - 処理完了後、「表示」ボタンで3Dモデルを確認
   - 「ダウンロード」ボタンで.glbファイルをダウンロード

#### APIをプログラムから使用する場合

```python
import requests

# 画像をアップロード
files = [('files', ('car1.jpg', open('car1.jpg', 'rb'), 'image/jpeg'))]
response = requests.post('http://localhost:8000/api/upload', files=files)
job_id = response.json()['job_id']

# パイプラインを実行
config = {
    'image_size': 1024,
    'bg_color': 'white',
    'mesh_method': 'poisson',
    'animation_type': 'orbit'
}
response = requests.post(f'http://localhost:8000/api/pipeline/{job_id}/start', json=config)

# 状態を確認
response = requests.get(f'http://localhost:8000/api/jobs/{job_id}')
status = response.json()
print(f"Status: {status['status']}, Progress: {status['progress']}%")
```

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

### 使用可能なパラメータ

```bash
python3 scripts/run_pipeline.py \
    --input_dir ./input \
    --output_dir ./output \
    --step all \
    --mesh_method poisson \
    --mesh_depth 10 \
    --mesh_resolution 256 \
    --mesh_smooth True \
    --texture_size 2048 \
    --specular_strength 0.5 \
    --roughness 0.3 \
    --metallic 0.1 \
    --clearcoat 0.5 \
    --source_images ./input
```

| パラメータ | 説明 | 既定値 |
|-----------|------|--------|
| `--mesh_method` | メッシュ化手法 | `poisson` |
| `--mesh_depth` | Poisson再構築の深さ | `10` |
| `--mesh_resolution` | メッシュ解像度 | `256` |
| `--mesh_smooth` | メッシュ平滑化 | `True` |
| `--texture_size` | テクスチャ解像度 | `2048` |
| `--specular_strength` | 鏡面反射強度 | `0.5` |
| `--roughness` | 粗さ（0.0 = 滑らか） | `0.3` |
| `--metallic` | 金属度（0.0 = 非金属） | `0.1` |
| `--clearcoat` | クリアコート量（車のクリアー層） | `0.5` |
| `--source_images` | テクスチャ生成用の元画像ディレクトリ | `None` |

### [`scripts/preprocess.py`](scripts/preprocess.py:1)
画像の前処理：
- 画像の正規化
- 車体のマスク生成（背景除去）
- 画像の整列

### [`scripts/colmap.py`](scripts/colmap.py:1)
COLMAP処理：
- Structure-from-Motion (SfM)
- カメラパラメータの推定
- 疎点群（Sparse Point Cloud）生成
- 画像間の対応点を検出

### [`scripts/gaussian_splatting.py`](scripts/gaussian_splatting.py:1)
Gaussian Splatting処理：
- 3Dガウス分布の最適化
- 高品質なレンダリング
- テクスチャの高精度な表現
- 反射・光沢の再現

### [`scripts/meshing.py`](scripts/meshing.py:1)
メッシュ化：
- **Poisson Surface Reconstruction**（Open3D使用）
  - 点群から滑らかなサーフェスを構築
  - depthパラメータで詳細度を制御（デフォルト: 10）
  - メッシュの簡素化（quadric decimation）
- **Instant Meshes**（Quad-Dominant）
  - 四辺形優位のメッシュを生成
  - 車のトポロジー最適化
- **DMVer2**（Depth-based Meshing）
  - 深度マップからのメッシュ生成
  - ポイントクオリティが低い場合のフォールバック
- **メッシュ平滑化**
  - Laplacian平滑化（イテレーション数、lambda値を指定可能）
- **出力形式**
  - GLB（バイナリGLTF、YouTube向け）
  - OBJ（オープンフォーマット）
  - PLY（ポイントクラウド/メッシュ）

### [`scripts/texture_baking.py`](scripts/texture_baking.py:1)
テクスチャベイク：
- **UV展開**
  - Angle-based unwrapping（角度保存）
  - LSCM（最小二乗 conformal mapping）
  - Morton orderingに基づく投影
- **テクスチャマッピング**
  - ポリゴンカラーからテクスチャをラスタライズ
  - 複数画像からの投影（マルチビューステレオ）
- **PBRマテリアル**
  - Base Color/Albedo map
  - Normal map
  - Roughness/Metallic maps
  - Clearcoat map（車のクリアークラース再現）
  - Specular map
- **光沢・反射の処理**
  - Specular strength（0.0 - 1.0）
  - Roughness（0.0 = 滑らか, 1.0 = 粗い）
  - Metallic（0.0 = 非金属, 1.0 = 金属）
  - Clearcoat（車のクリアーコート層）

### [`scripts/blender_video.py`](scripts/blender_video.py:1)
Blender動画生成：
- 複数の3Dモデルを比較
- YouTube向けの動画を作成
- アニメーションの追加

### [`scripts/run_pipeline.py`](scripts/run_pipeline.py:1)
メインパイプラインスクリプト：
- 全体フローの制御
- 各ステップのオーケストレーション
- 単一ステップの実行も可能

## テスト

### テストファイル構成

```
tests/
├── __init__.py            # テストパッケージ
├── conftest.py            # 共通フィクスチャ
├── test_preprocess.py     # 前処理テスト
├── test_colmap.py         # COLMAPテスト
├── test_gaussian_splatting.py   # Gaussian Splattingテスト
├── test_meshing.py        # メッシュ化テスト
├── test_texture_baking.py     # テクスチャベイクテスト
├── test_blender_video.py      # Blender動画生成テスト
├── test_run_pipeline.py     # パイプラインテスト
└── test_app_main.py       # Web APIテスト
```

### テストの実行方法

```bash
# すべてのテストを実行
python -m pytest tests/ -v

# 特定モジュールのテストのみを実行
python -m pytest tests/test_preprocess.py -v
python -m pytest tests/test_colmap.py -v
python -m pytest tests/test_meshing.py -v

# カバレッジ付きで実行
python -m pytest tests/ -v --cov=scripts --cov=app --cov-report=html

# クイックテスト（マーク付き）
python -m pytest tests/ -v -m unit

# Dockerコンテナ内でテストを実行
docker compose exec car-model pytest tests/ -v
```

### テストカバレッジ対象モジュール

| モジュール | テストファイル | 主要内容 |
|-----------|---------------|----------|
| [`scripts/preprocess.py`](scripts/preprocess.py) | [`tests/test_preprocess.py`](tests/test_preprocess.py) | 画像正規化、背景除去、アライメント |
| [`scripts/colmap.py`](scripts/colmap.py) | [`tests/test_colmap.py`](tests/test_colmap.py) | カメラ推定、点群読み込み、PLY形式 |
| [`scripts/gaussian_splatting.py`](scripts/gaussian_splatting.py) | [`tests/test_gaussian_splatting.py`](tests/test_gaussian_splatting.py) | 空間設定、設定生成、合成出力 |
| [`scripts/meshing.py`](scripts/meshing.py) | [`tests/test_meshing.py`](tests/test_meshing.py) | ポアソン再構築、メッシュ出力、平滑化 |
| [`scripts/texture_baking.py`](scripts/texture_baking.py) | [`tests/test_texture_baking.py`](tests/test_texture_baking.py) | UV展開、マテリアル、テクスチャ出力 |
| [`scripts/blender_video.py`](scripts/blender_video.py) | [`tests/test_blender_video.py`](tests/test_blender_video.py) | モデル探索、Blenderスクリプト生成 |
| [`scripts/run_pipeline.py`](scripts/run_pipeline.py) | [`tests/test_run_pipeline.py`](tests/test_run_pipeline.py) | パイプライン各ステップ |
| [`app/main.py`](app/main.py) | [`tests/test_app_main.py`](tests/test_app_main.py) | FastAPIエンドポイント、Pydanticモデル |

### 共通フィクスチャ（conftest.py）

テストで使用する共通のフィクスチャ:

- `temp_dir`: 一時的ディレクトリ
- `sample_image_dir`: サンプル画像ディレクトリ
- `sample_colmap_dir`: COLMAP出力ディレクトリ
- `sample_gs_output_dir`: Gaussian Splatting出力ディレクトリ
- `sample_mesh_dir`: メッシュ出力ディレクトリ
- `sample_models_dir`: 3Dモデルディレクトリ
- `colmap_points_data`: COLMAP points3D.binバイナリデータ
- `ascii_ply_content`: ASCII PLYファイル内容
- `mesh_data`: サンプルメッシュデータ

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
| [`docker-compose.yml`](docker-compose.yml) | Docker Compose設定（GPU対応、FastAPI対応） |
| [`.dockerignore`](.dockerignore) | Docker除外ファイル |
| [`build.sh`](build.sh) | ビルドスクリプト |
| [`run_car_model.sh`](run_car_model.sh) | 実行スクリプト |
| [`.clinerules`](.clinerules) | プロジェクトルール |
| [`app/__init__.py`](app/__init__.py) | アプリケーションパッケージ |
| [`app/main.py`](app/main.py) | FastAPIサーバー（Web API） |
| [`scripts/preprocess.py`](scripts/preprocess.py) | 前処理スクリプト |
| [`scripts/colmap.py`](scripts/colmap.py) | COLMAP処理スクリプト |
| [`scripts/gaussian_splatting.py`](scripts/gaussian_splatting.py) | Gaussian Splattingスクリプト |
| [`scripts/meshing.py`](scripts/meshing.py) | メッシュ化スクリプト |
| [`scripts/texture_baking.py`](scripts/texture_baking.py) | テクスチャベイクスクリプト |
| [`scripts/blender_video.py`](scripts/blender_video.py) | Blender動画生成スクリプト |
| [`scripts/run_pipeline.py`](scripts/run_pipeline.py) | メインパイプラインスクリプト |

## ライセンス

このプロジェクトはMITライセンスの下で提供されます。

## 作者

LGM Win 3090 Project
