# Overview

evprtr is a local **composite layer**: it calls existing model runtimes and combines them with inference-time mechanisms so everyday agent work (the kind done with strong cloud models) can stay on-machine.

Point OpenCode, Codex, Pi, or similar harnesses at evprtr’s OpenAI-compatible endpoint. Under the hood, the compositor may use Maple-Preview for reasoning, Needle for reliable tool/JSON contracts, and verification loops when something fails.

See [architecture.md](./architecture.md) for layers, naming (including why the project is called *evprtr*), and design boundaries.
