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
