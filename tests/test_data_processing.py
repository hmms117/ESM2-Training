from src import data_processing


def test_pad_batch_basic():
    batch = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [5, 6]},
        {"input_ids": [3], "attention_mask": [1], "labels": [7]},
    ]
    padded = data_processing.pad_batch(batch, max_length=3, pad_token_id=0)
    for ex in padded:
        assert len(ex["input_ids"]) == 3
        assert len(ex["attention_mask"]) == 3
        assert len(ex["labels"]) == 3
    assert padded[1]["input_ids"] == [3, 0, 0]
    assert padded[1]["labels"][-1] == -100


def test_refine_fasta(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(
        ">seq1\nABCDE\n>seq2\nABCDEFG\n>seq1\nXYZ\n>seq3\nABC\n"
    )
    output = tmp_path / "refined.fasta"
    result = data_processing.refine_fasta(str(fasta), str(output), max_length=5)
    assert ">seq1" in result and result[">seq1"] == "ABCDE"
    assert ">seq3" in result and result[">seq3"] == "ABC"
    assert ">seq2" not in result  # too long
    assert output.exists()
    text = output.read_text()
    assert ">seq2" not in text
    assert text.count("seq1") == 1
