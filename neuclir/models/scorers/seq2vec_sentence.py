import torch
import torch.nn as nn

from .scorer import Scorer
from typing import Optional
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.attention import Attention
from allennlp.nn import Activation

@Scorer.register("s2v_sentence_scorer")
class Seq2VecSentenceScorer(Scorer):
    """
    The ``Seq2VecScorer'' is a ``Scorer'' which computes a single vector
    representation for each query and document, and then computes the

    Parameters
    ----------
    doc_encoder: ``Seq2VecEncoder''
        This is the encoder for the document. Common options include the
        ``BagOfEmbeddingsEncoder'' or the ``CNNEncoder''.
    query_encoder: ``Seq2VecEncoder''
        This is the encoder for the query. Can be of the same type or a
        different type than that of the document encoder.
    use_encoded: ``bool'', optional (default = ``False'')
        Whether or not to use the encoded representations.
    scorer: ``FeedForward'', optional (default = ``None'')
        If no feedforward scorer is provided, we use a single-layer feedforward
        network that uses.
    attention: ``Attention'', optional (default = ``None'')
        If attention is provided, we use it to compute a query-specific document
        representation.
    """

    def __init__(self,
                 sentence_encoder: Seq2VecEncoder,
                 doc_encoder: Seq2VecEncoder,
                 query_encoder: Seq2VecEncoder,
                 use_encoded: bool = False,
                 scorer: Optional[FeedForward] = None,
                 sentence_attention: Optional[Attention] = None,
                 document_attention: Optional[Attention] = None) -> None:

        super(Seq2VecSentenceScorer, self).__init__()

        self.sentence_encoder = sentence_encoder
        self.doc_encoder = doc_encoder
        self.query_encoder = query_encoder
        self.use_encoded = use_encoded
        self.sentence_attention = sentence_attention
        self.document_attention = document_attention
        # get the dimensions for the scorer and for sanity checking
        q_dim = self.query_encoder.get_output_dim()
        d_dim = self.doc_encoder.get_output_dim()

        input_dim = (q_dim + d_dim)
        if use_encoded: input_dim *= 2
        # set up the scorer
        if scorer is None:
            scorer = FeedForward(
                        input_dim=input_dim, num_layers=1,
                        hidden_dims=1, activations=Activation.by_name('linear')(), dropout=0.)
        self.scorer = scorer
        # assertions to ensure our shapes match our assumptions
        assert q_dim == d_dim
        assert self.scorer.get_output_dim() == 1
        assert self.scorer.get_input_dim() == input_dim

    def forward(self,
                query: torch.Tensor,
                document: torch.Tensor,
                query_mask: torch.Tensor = None,
                sentence_mask: torch.Tensor = None,
                document_mask: torch.Tensor = None) -> torch.Tensor:

        # (batch_size, query_length, embedding_dim)
        q = self.query_encoder(query, query_mask)

        batch_size, num_sentences, sentence_length, embedding_dim = document.shape
        # (batch_size*num_sentences, embedding_dim)
        sentences = document.view(batch_size*num_sentences, sentence_length, embedding_dim)
        sentence_mask = sentence_mask.view(batch_size*num_sentences, sentence_length)

        if self.sentence_attention is not None:
            # (batch_size, embedding_dim)
            # (batch_size, num_sentences, embedding_dim)
            q_sentences = q.unsqueeze(1).repeat(1, num_sentences, 1)
            # (batch_size*num_sentences, embedding_dim)
            q_sentences = q_sentences.view(batch_size*num_sentences, embedding_dim)
            attn = self.sentence_attention(q_sentences, sentences, sentence_mask).unsqueeze(2)
            sentences = sentences * attn

        # (batch_size*num_sentences, encoding_dim)
        sentences_encoded = self.sentence_encoder(sentences, sentence_mask)
        # (batch_size, num_sentences, encoding_dim)
        sentences_encoded = sentences_encoded.view(batch_size, num_sentences, -1)

        if self.document_attention is not None:
            attn = self.document_attention(q, sentences_encoded, document_mask).unsqueeze(2)
            sentences_encoded = sentences_encoded * attn

        docs_encoded = self.doc_encoder(sentences_encoded, document_mask)

        reprs = [q - docs_encoded, q * docs_encoded]
        if self.use_encoded:
            reprs += [q, docs_encoded]

        encoded = torch.cat(reprs, dim=1)

        return self.scorer(encoded)
