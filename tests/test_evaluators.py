from fairlens.evaluators.stereotype import StereotypeDetector


def test_stereotype_detect_hit():
    det = StereotypeDetector({"man": ["aggressive", "dominant"]})
    result = det.detect("He was very aggressive and loud", "man")
    assert result["stereotype_hit"] is True
    assert "aggressive" in result["matches"]


def test_stereotype_detect_miss():
    det = StereotypeDetector({"man": ["aggressive"]})
    result = det.detect("He was kind and thoughtful", "man")
    assert result["stereotype_hit"] is False
    assert result["matches"] == []


def test_stereotype_no_lexicon():
    det = StereotypeDetector()
    result = det.detect("anything here", "unknown_group")
    assert result["stereotype_hit"] is False


def test_stereotype_rate():
    det = StereotypeDetector({"x": ["bad"]})
    texts = ["this is bad", "this is good", "also bad stuff", "fine"]
    rate = det.stereotype_rate(texts, "x")
    assert rate == 0.5


def test_stereotype_rate_empty():
    det = StereotypeDetector({"x": ["bad"]})
    assert det.stereotype_rate([], "x") == 0.0


def test_stereotype_word_boundary():
    det = StereotypeDetector({"g": ["art"]})
    result = det.detect("He started the art project", "g")
    assert result["stereotype_hit"] is True

    result2 = det.detect("He is smart", "g")
    assert result2["stereotype_hit"] is False


def test_set_lexicon():
    det = StereotypeDetector()
    det.set_lexicon({"a": ["Word"]})
    result = det.detect("this word appears", "a")
    assert result["stereotype_hit"] is True
