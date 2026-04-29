"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/gpu_utils.py
Reservoir Computing用のGPUユーティリティ関数。
"""
from __future__ import annotations

import re
import subprocess
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class JaxDeviceLike(Protocol):
    """Minimal JAX device surface used by this module."""
    id: int
    platform: str
    device_kind: str


def _device_text(device: JaxDeviceLike) -> str:
    return f"{device.platform}:{device.device_kind}:{device.id}"


def _assert_x64_enabled() -> None:
    """Raise if JAX x64 mode is not enabled."""
    import jax
    if not jax.config.read("jax_enable_x64"):
        raise ValueError("CRITICAL: JAX x64 mode is NOT enabled. Double-check import order and environment variables.")


def _assert_gpu_detected(gpu_devices: Sequence[JaxDeviceLike], devices: Sequence[JaxDeviceLike]) -> None:
    """Raise if no GPU devices are found."""
    if not gpu_devices:
        print("ERROR: GPU not detected!")
        print("Available devices:", [_device_text(d) for d in devices])
        if devices and all(d.platform.lower() == "cpu" for d in devices):
            print("Only CPU devices detected - GPU initialization failed")
        raise RuntimeError(
            "GPU not detected. This is a GPU-required test.\n"
            "詳細なトラブルシューティングは docs/TROUBLESHOOTING.md を参照してください\n"
            "または --force-cpu オプションでCPU実行も可能です"
        )


def check_gpu_available() -> bool:
    """JAXがGPUを認識し、利用可能かを確認。
    
    JAXが利用可能なデバイスリストを取得し、その中にGPUが含まれているかを
    検証します。簡単な計算テストも実行し、GPUが正常に動作することを
    確認します。
    
    Returns:
        GPUが利用可能な場合はTrue
    
    Raises:
        RuntimeError: GPUが検出されない、または計算テストに失敗した場合
    """
    import jax
    import jax.numpy as jnp
    
    print("=== GPU認識確認 ===")

    try:
        # Enforce x64 explicitly before checking/initializing backend
        jax.config.update("jax_enable_x64", True)
        _assert_x64_enabled()

        x64_enabled = jax.config.read("jax_enable_x64")
        print(f"JAX x64 Enabled: {x64_enabled}")
        
        devices = jax.devices()
        print(f"JAXバージョン: {jax.__version__}")

        # GPU利用可能性チェック
        gpu_devices = [d for d in devices if d.platform.lower() == "gpu" or d.platform.lower() == "cuda"]
        _assert_gpu_detected(gpu_devices, devices)

        # 簡単なGPU計算テスト
        try:
            # Force float64 creation to test if it's respected
            x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
            print(f"Test Array Dtype: {x.dtype.name}")
            
            if x.dtype != jnp.float64:
                 print("WARNING: JAX did not create float64 array despite request. x64 mode might be failed.")
            
            result = jnp.sum(x ** 2)

            # GPU名を取得（利用可能な場合）
            device = gpu_devices[0]
            driver_version = None
            cuda_version = None
            nvcc_version = None
            try:
                # nvidia-mlまたはpynvmlを使ってGPU名とバージョン情報を取得
                # GPU名
                nvidia_smi_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip().split('\n')[0]
                device_name = nvidia_smi_output

                # Driver / CUDA version を取得
                smi_full_output = subprocess.check_output(
                    ["nvidia-smi"],
                    text=True,
                )
                version_match = re.search(
                    r"Driver Version:\s*([\d.]+)\s+CUDA Version:\s*([\d.]+)",
                    smi_full_output
                )
                if version_match:
                    driver_version, cuda_version = version_match.groups()

                try:
                    nvcc_output = subprocess.check_output(
                        ["nvcc", "--version"],
                        text=True,
                    )
                    nvcc_match = re.search(r"release\s+([\d.]+)", nvcc_output)
                    if nvcc_match:
                        nvcc_version = nvcc_match.group(1)
                except (OSError, subprocess.SubprocessError):
                    nvcc_version = None
            except (OSError, subprocess.SubprocessError):
                # nvidia-smiが失敗した場合はデバイスIDのみ表示
                device_name = f"CUDA Device {device.id}"

            print(f"GPU detected: {device_name}")
            if driver_version:
                print(f"Driver Version: {driver_version}")
            if cuda_version:
                print(f"CUDA Version: {cuda_version}")
            if nvcc_version:
                print(f"nvcc CUDA Toolkit: {nvcc_version}")
            print(f"GPU計算テスト成功: {float(result)}")
        except RuntimeError as e:
            print(f"GPU計算テスト失敗: {e}")
            raise RuntimeError(f"GPU computation test failed: {e}") from e
        else:
            return True

    except RuntimeError as e:
        print(f"GPU確認エラー: {e}")
        raise


def require_gpu() -> Callable[[Callable[[], None]], Callable[[], None]]:
    """GPUを必須とするテスト用のデコレータ。
    
    このデコレータを付けたテスト関数は、実行前に`check_gpu_available`を
    呼び出します。GPUが利用できない場合、テストはスキップされず、
    エラーとして終了します。
    
    Returns:
        デコレートされたテスト関数ラッパー
        
    Examples:
        >>> @require_gpu()
        ... def test_gpu_specific_feature():
        ...     # GPUがなければこのテストは実行されない
        ...     assert True
    """
    import sys
    
    def decorator(test_func: Callable[[], None]) -> Callable[[], None]:
        def wrapper() -> None:
            try:
                check_gpu_available()
                test_func()
            except RuntimeError as e:
                print(f"\nGPU REQUIREMENT FAILED: {e}")
                print("Exiting test due to GPU requirement...")
                sys.exit(1)
        return wrapper
    return decorator


def print_gpu_info() -> None:
    """現在JAXが使用しているGPUデバイス情報を表示。
    
    JAXが認識しているデバイスリストからGPUデバイスを特定し、
    その情報を標準出力に表示します。GPUが利用できない場合は
    その旨を伝えます。
    
    Examples:
        >>> print_gpu_info()
        Using GPU: T4
    """
    import jax
    
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform.lower() == "gpu" or d.platform.lower() == "cuda"]
        
        if gpu_devices:
            print(f"Using GPU: {gpu_devices[0]}")
        else:
            print("No GPU available - using CPU")
            
    except RuntimeError as e:
        print(f"Cannot get GPU info: {e}")
