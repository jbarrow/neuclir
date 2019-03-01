from typing import List
from allennlp.data.tokenizers import Token

def tokenize(line: str) -> List[Token]:
    return [Token(word) for word in line.split()]
