import jax
import jax.numpy as jnp
from reservoir.models.reservoir.quantum.model import QuantumReservoir
from reservoir.layers.aggregation import AggregationMode


def measure_circuit_attenuation():
    # 1. 今の設定でモデルを作成（設定ファイルと同じ条件にする）
    # ※特に n_layers, encoding_strategy, measurement_basis が重要
    model = QuantumReservoir(
        n_qubits=16,  # 現在の設定
        n_layers=4,  # 現在の設定
        seed=42,
        aggregation_mode=AggregationMode.SEQUENCE,
        feedback_scale=1.0,  # ここは測定用に1.0固定（基準）
        leak_rate=1.0,
        measurement_basis="Z+ZZ",
        noise_type="clean",
        noise_prob=0.0,
        readout_error=0.0,
        n_trajectories=0,
        use_reuploading=True,  # 現在の設定
    )

    print("--- Circuit Attenuation Test ---")
    print(f"Config: {model.n_qubits} Qubits, {model.n_layers} Layers, Re-uploading={model.use_reuploading}")

    # 2. テストデータの生成
    # フィードバックループに入力されると仮定したランダム信号
    # 標準正規分布 (Mean=0, Std=1.0)
    n_samples = 2000

    # 状態としての入力 (Feedbackとして入る値)
    # 実際のフィードバックは「前の時刻の出力(Z+ZZ)」が「次の回転角」になる
    # 入力次元は n_qubits (16) ではなく、feedback_size (16 or 136)
    # ここでは簡易的に「回路を通した時の信号強度の比率」を見ます

    # ランダムな状態ベクトルを生成して回路を通す実験
    rng = jax.random.key(0)
    state = model.initialize_state(n_samples)  # |0>

    # 強制的に「ある強さの入力」を注入して、出力がどうなるか見る
    # Input Scaleの影響も受けるため、実質的なゲインを測る

    # 入力信号の強さ (分散) を変えてテスト
    test_scales = [1.0]

    for scale in test_scales:
        # ランダムな入力を生成 (これが前の時刻の記憶に相当)
        rng, k = jax.random.split(rng)
        random_feedback = jax.random.normal(k, (n_samples, 16)) * scale

        # モデルのステップ関数を実行
        # input_data=0 にして、純粋に feedback (random_feedback) だけの影響を見る
        # step関数内部: input + feedback_scale * state
        # ここでは擬似的に input_data として random_feedback を渡して挙動を見る

        # ※ 正確には QuantumReservoir.step は (state, input) を受け取る。
        # 今回の知りたい「減衰」は「入力された回転角 θ が、出力 Z になるときにどれくらい縮むか」

        # 1層目の回転角 θ_in ≈ random_feedback となる状況を作る
        # input_scale=1.0 と仮定

        _, output = model.step(state, random_feedback)

        # 統計量の計算
        input_std = float(jnp.std(random_feedback))
        output_std = float(jnp.std(output))

        attenuation_ratio = output_std / input_std
        recommended_feedback = 1.0 / attenuation_ratio if attenuation_ratio > 0 else 999.0

        print(f"\n[Input Std: {input_std:.4f}]")
        print(f"  -> Output Std: {output_std:.4f}")
        print(f"  -> Attenuation Factor (α): {attenuation_ratio:.4f}")
        print(f"  -> Theoretical Optimal Feedback (1/α): {recommended_feedback:.4f}")


if __name__ == "__main__":
    measure_circuit_attenuation()
