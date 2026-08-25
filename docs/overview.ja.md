> Source: overview.md @ uncommitted

# 概要

evprtr はローカル向けの **composite layer（合成層）** です。既存のモデル runtime を呼び出し、推論時の機構と組み合わせることで、強いクラウドモデルで行うような日常のエージェント作業をマシン内で完結しやすくします。

OpenCode / Codex / Pi などの harness からは、OpenAI 互換エンドポイントとして見えます。内部では推論に Maple-Preview、tool / JSON の契約遵守に Needle、失敗時の検証・修復ループなどを用いる想定です。

層の定義・命名（プロジェクト名 *evprtr* の由来を含む）・設計境界は [architecture.md](./architecture.md) を参照してください。
