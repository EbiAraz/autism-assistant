from data_loader import _merge_fields


def test_merge_fields_combines_non_empty_values():
    row = {"instruction": "Speak clearly", "input": "", "output": "Response"}
    text = _merge_fields(row, ("instruction", "input", "output"))

    assert "instruction: Speak clearly" in text
    assert "output: Response" in text
    assert "input:" not in text


def test_merge_fields_returns_empty_for_no_text():
    row = {"instruction": "", "input": None, "output": "  "}
    assert _merge_fields(row, ("instruction", "input", "output")) == ""
