from llm_from_scratch.tokenizer import ByteTokenizer

def test_round_trip_unicode():
    t=ByteTokenizer(); s="Hello, جهان 👋"; assert t.decode(t.encode(s))==s
