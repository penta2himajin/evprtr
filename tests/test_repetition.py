from compositor.verify.repetition import find_repetition, truncate_before_repetition


def test_detects_word_run_like_maple_collapse():
    text = "The harness is the agent " + ("agent " * 20) + "layer."
    hit = find_repetition(text)
    assert hit is not None
    assert hit.kind in {"word_run", "char_motif", "ngram_run"}
    truncated = truncate_before_repetition(text, hit)
    assert truncated.startswith("The harness is the")
    assert truncated.count("agent") < text.count("agent")


def test_detects_char_motif_like_evprtrtrtr():
    text = "Project name is evpr" + ("tr" * 40) + " and more."
    hit = find_repetition(text)
    assert hit is not None
    assert hit.kind in {"char_motif", "word_run", "ngram_run"}


def test_healthy_text_passes():
    text = (
        "evprtr is a composite layer between harnesses and runtimes. "
        "It calls local models and does not distill weights. "
        "The name comes from a maple sap evaporator."
    )
    assert find_repetition(text) is None


def test_short_text_skipped():
    assert find_repetition("agent agent agent") is None


def test_char_motif_ignores_whitespace_only_motifs():
    # Markdown/code indentation is idiomatic, not model collapse: the live
    # paired measurement fired char_motif "  " x10 on a healthy Python code
    # block (20-space indent before a comment) and evprtr truncated the
    # answer (1899 -> 1395 chars for the same upstream output).
    text = (
        "Recursion is a function calling itself.\n\n"
        "```python\n"
        "def factorial(n):\n"
        "    if n == 0:               # base case\n"
        "        return 1\n"
        "    else:\n"
        "                    # recursive case\n"
        "        return n * factorial(n - 1)\n"
        "```\n"
        "The base case stops the recursion."
    )
    hit = find_repetition(text)
    assert hit is None, f"healthy indentation must not fire, got {hit}"


def test_char_motif_still_detects_non_whitespace_motifs():
    text = "processing complete: " + ("tr" * 20) + " end"
    hit = find_repetition(text)
    assert hit is not None
    assert hit.kind == "char_motif"
    assert hit.detail["motif"].strip() != ""
