> Source: README.md @ uncommitted

# evprtr

[English](./README.md)

ローカルのモデル **runtime を呼び出す composite layer（合成層）**。OpenAI 互換の一つのモデル面として出し、OpenCode / Codex / Pi などの harness から、フロンティア級クラウドモデルに頼らず実用的なエージェント作業を進められるようにする。

名前 **evprtr** は、メープルシロップ製造で樹液を煮詰める **evaporator（蒸発器）** に由来する。能力の濃縮は蒸留ではなく機構で行う。初期の中心は Maple-Preview と、Needle などの補助。詳細は [docs/architecture.md](./docs/architecture.md)。

## 現状

OpenAI 互換 API（`stream=true` SSE shim）、失敗トレース、verify/repair、Needle tool パス、compositor 側 buffer（path A）、**Pi harness ゲート**（path B: `harness/pi/`）まで。

## クイックスタート

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev,needle]"

set EVPRTR_UPSTREAM_BASE_URL=http://127.0.0.1:8080/v1
set EVPRTR_NEEDLE_LIB_PATH=F:\LLM\models\needle2\libneedle.dll
python -m compositor

# harness の base URL: http://127.0.0.1:8741/v1  （model id: evprtr）
```

```bash
pytest
```

## レイアウト

| パス | 役割 |
|---|---|
| `compositor/` | composite layer 本体（API + オーケストレーション） |
| `tests/` | TDD スイート |
| `docs/architecture.md` | 層・命名・境界 |
| `docs/overview.md` | 短い概要 |

## ライセンス

MIT。`LICENSE` を参照。
