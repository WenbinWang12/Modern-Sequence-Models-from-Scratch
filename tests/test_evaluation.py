"""召回评测的输入边界。"""

from ttt.data import ByteTokenizer
from ttt.eval_retrieval import evaluate_retrieval


def test_answers_are_only_used_for_scoring():
    tokenizer = ByteTokenizer()

    class Session:
        prompt = None

        def reset(self):
            self.prompt = None

        def generate(self, tokens, **kwargs):
            self.prompt = tokenizer.decode(tokens)
            return tokenizer.encode("unseen-answer", add_bos=False)

    session = Session()
    result = evaluate_retrieval(session, tokenizer, [{"prompt": "query only", "answer": "unseen-answer"}])
    assert session.prompt == "query only"
    assert result["exact_match"] == 1.0
