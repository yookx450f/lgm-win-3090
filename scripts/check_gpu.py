#!/usr/bin/env python3
"""
GPUチェックスクリプト - RTX 3090が正常に動作しているか確認

このスクリプトは以下を確認します：
1. ホストOSでのNVIDIA GPUアクセス
2. Dockerコンテナ内のGPUアクセス
3. PyTorchのCUDAサポート
"""

import subprocess
import sys
import os


def check_host_gpu():
    """ホストOSでのGPUアクセスを確認"""
    print("=" * 60)
    print("  ホストOS GPUチェック")
    print("=" * 60)
    
    # nvidia-smiチェック
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  [SUCCESS] nvidia-smi 動作確認完了")
            print("\n  GPU情報:")
            # 出力からGPU名とメモリ使用量のみ抽出
            lines = result.stdout.split('\n')
            for line in lines[:20]:  # 最初の20行のみ表示
                if 'Default' in line or 'N/A' in line or 'Process' in line:
                    continue
                if line.strip() and not line.startswith('+'):
                    print(f"    {line}")
        else:
            print("  [ERROR] nvidia-smi が動作しません")
            print("  解決策:")
            print("    1. NVIDIAドライバーがインストールされているか確認: ls /dev/nvidia*")
            print("    2. WSL2カーネルが最新か確認: wsl --update")
            print("    3. GPUが有効か確認: wslcat -e 'cat /proc/driver/nvidia/version'")
            return False
    except FileNotFoundError:
        print("  [ERROR] nvidia-smi が見つかりません")
        print("  解決策: NVIDIAドライバーをインストールしてください")
        return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] nvidia-smi がタイムアウトしました")
        return False
    
    return True


def check_docker_gpu():
    """Dockerコンテナ内のGPUアクセスを確認"""
    print("\n" + "=" * 60)
    print("  Docker GPUチェック")
    print("=" * 60)
    
    # DockerがGPUを認識しているか確認
    try:
        result = subprocess.run(
            ['docker', 'run', '--rm', 'nvidia/cuda:13.0.0-devel-ubuntu22.04', 
             'nvidia-smi'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("  [SUCCESS] Dockerコンテナ内でnvidia-smiが動作します")
            # GPU名を表示
            for line in result.stdout.split('\n'):
                if 'RTX' in line or 'NVIDIA' in line:
                    print(f"    {line.strip()}")
        else:
            print("  [ERROR] Dockerコンテナ内でnvidia-smiが動作しません")
            print("  解決策:")
            print("    1. NVIDIA Container Toolkitがインストールされているか確認")
            print("    2. Docker設定でGPUが有効か確認: /etc/docker/daemon.json")
            print("    3. Dockerを再起動: sudo systemctl restart docker")
            return False
    except FileNotFoundError:
        print("  [ERROR] docker が見つかりません")
        print("  解決策: Dockerをインストールしてください")
        return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] Dockerコマンドがタイムアウトしました")
        return False
    
    return True


def check_pytorch_cuda():
    """PyTorchのCUDAサポートを確認"""
    print("\n" + "=" * 60)
    print("  PyTorch CUDAチェック")
    print("=" * 60)
    
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA Version (compiled): {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}:")
                print(f"    Name: {torch.cuda.get_device_name(i)}")
                print(f"    Capability: {torch.cuda.get_device_capability(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Total Memory: {props.total_mem_mb / 1024:.1f} MB")
                print(f"    Multi-Processor Count: {props.multi_processor_count}")
            
            # テスト演算
            print("\n  テスト演算:")
            try:
                test_tensor = torch.ones(1000, 1000, device='cuda')
                test_result = torch.sum(test_tensor)
                print(f"    [SUCCESS] CUDA演算正常動作 (結果: {test_result.item()})")
                del test_tensor, test_result
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    [ERROR] CUDA演算失敗: {e}")
                return False
        else:
            print("\n  [WARNING] CUDAが利用できません!")
            print("  考えられる原因:")
            print("    1. PyTorchがCUDAなしでインストールされている")
            print("       再インストール: pip install torch --extra-index-url https://download.pytorch.org/whl/cu124")
            print("    2. NVIDIAドライバーのバージョンが古い")
            print("    3. CUDAツールキットのパスが設定されていない")
            return False
        
        return True
        
    except ImportError:
        print("  [ERROR] PyTorchがインストールされていません")
        print("  解決策: pip install torch --extra-index-url https://download.pytorch.org/whl/cu124")
        return False


def check_docker_compose_config():
    """docker-compose.ymlの設定を確認"""
    print("\n" + "=" * 60)
    print("  docker-compose.yml 設定チェック")
    print("=" * 60)
    
    compose_files = [
        'docker-compose.yml',
        'docker-compose.yaml',
    ]
    
    found = False
    for compose_file in compose_files:
        if os.path.exists(compose_file):
            found = True
            print(f"  ファイル: {compose_file}")
            
            with open(compose_file, 'r') as f:
                content = f.read()
                
            checks = {
                'runtime: nvidia': 'NVIDIA runtime設定',
                'NVIDIA_VISIBLE_DEVICES': 'GPU可視性設定',
                'NVIDIA_DRIVER_CAPABILITIES': 'ドライバー機能設定',
                '/dev/dri': 'DRIデバイスマッピング',
                '/dev/nvidia0': 'NVIDIAデバイスマッピング',
            }
            
            for check, name in checks.items():
                if check in content:
                    print(f"    [OK] {name}")
                else:
                    print(f"    [MISSING] {name}")
            
            # deploy.resources.reservationsがある場合は警告
            if 'deploy:' in content and 'reservations:' in content:
                print("\n  [WARNING] 従来のdeploy方式が使用されています")
                print("    推奨: runtime: nvidia + devices: 方式")
            
            break
    
    if not found:
        print("  [ERROR] docker-composeファイルが見つかりません")
        return False
    
    return True


def main():
    """メイン処理"""
    print("\\n" + "##############################################")
    print("#  LGM Car Model - GPUチェックツール")
    print("#  RTX 3090動作確認")
    print("##############################################\\n")
    
    results = {
        'ホストOS GPU': check_host_gpu(),
        'Docker GPU': check_docker_gpu(),
        'PyTorch CUDA': check_pytorch_cuda(),
        'docker-compose設定': check_docker_compose_config(),
    }
    
    print("\\n" + "=" * 60)
    print("  chk結果")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("")
    if all_passed:
        print("  [SUCCESS] 全てのチェックが通りました！")
        print("  RTX 3090が正常に動作しています。")
    else:
        print("  [WARNING] 一部のチェックに失敗しました。")
        print("  上記の解決策をお試しください。")
    
    print("")
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
