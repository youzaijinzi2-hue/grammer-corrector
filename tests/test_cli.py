from types import SimpleNamespace

from grammer_corrector import cli
from grammer_corrector.tokenizer import build_vocabulary, encode


def test_interactive_cli(monkeypatch, capsys):
    vocabulary = build_vocabulary(["He go outside.", "He goes outside."])
    embedding = SimpleNamespace(position=SimpleNamespace(pe=SimpleNamespace(shape=(1, 16))))
    model = SimpleNamespace(
        encoder=SimpleNamespace(embedding=embedding),
        decoder=SimpleNamespace(embedding=embedding),
    )
    loads = []

    def load(path):
        loads.append(path.name)
        return model, vocabulary

    calls = []

    def generate(model, source_ids, vocabulary, limit):
        calls.append(source_ids)
        return encode("He goes outside.", vocabulary)

    sentences = iter(["He go outside.", "He goes outside.", "He swims.", ":quit"])
    monkeypatch.setattr("sys.argv", ["cli", "model.pt"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(sentences))
    monkeypatch.setattr(cli, "load_checkpoint", load)
    monkeypatch.setattr(cli, "greedy_decode", generate)

    cli.main()

    assert loads == ["model.pt"]
    assert calls == [encode(text, vocabulary) for text in ("He go outside.", "He goes outside.")]
    assert capsys.readouterr().out.splitlines()[1:] == [
        "He goes outside.",
        "He goes outside.",
        "Unknown token. Returning the original sentence.",
        "He swims.",
    ]
