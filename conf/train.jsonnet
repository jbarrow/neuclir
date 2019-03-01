# EXTERNAL VARIABLE LIST
# dan: bool,
# averaged: bool,
# num_filters: int,
# dropout: float,
# batch_size: int,
# clipping: float,
# lr: float,
# l2: float,
# dataset: str

# Variables

local use_scores = true;

local embedding_dims = std.extVar('embedding_dims');
local dataset = std.extVar('dataset');
local lang = std.extVar('lang');
local idf_weights = std.extVar('idf_weights');
local dan = std.extVar('dan');
local doc_projection = std.extVar('doc_projection');
local averaged = std.extVar('averaged');
local num_filters = std.extVar('num_filters');
local query_averaged = std.extVar('query_averaged');
local l2 = std.extVar('l2');
local lr = std.extVar('lr');


local random_seed = 2019;
local pytorch_seed = random_seed * 10;
local numpy_seed = pytorch_seed * 10;

# Helper Functions
local Embedder(path, dim, trainable=false, projection=false) = {
  tokens: {
    type: 'embedding',
    pretrained_file: path,
    embedding_dim: dim,
    trainable: trainable,
  } + if projection then { projection_dim: embedding_dims } else { }
};

local EmbeddingTransformer(dim, dropout=0.5, activation='relu') = {
  input_dim: dim,
  num_layers: 1,
  hidden_dims: [dim],
  activations: [activation],
  dropout: [dropout]
};

local Scorer(embedding_dim, lexical_input=false) = {
  local lexical_dims = if lexical_input then 1 else 0,
  input_dim: embedding_dim * 2,# + lexical_dims,
  num_layers: 1,
  hidden_dims: [1],
  activations: ['sigmoid'],
  dropout: [0.0]
};

local doc_encoder = if dan then {
  type: 'boe',
  embedding_dim: embedding_dims,
  averaged: averaged
} else {
  type: 'cnn',
  embedding_dim: embedding_dims,
  num_filters: num_filters,
  output_dim: embedding_dims
};

local query_encoder = {
  type: 'boe',
  embedding_dim: embedding_dims,
  averaged: query_averaged
};

#local Pathify(relative_path) = '/storage3/proj/joe/neuclir/' + relative_path;
#local Pathify(relative_path) = '/fs/clip-scratch/jdbarrow/neuclir/neuclir/' + relative_path;
#local Pathify(relative_path) = '/storage2/proj/joe/neuclir/' + relative_path;
local Pathify(relative_path) = '/home/joe/projects/neuclir/' + relative_path;

{
  random_seed: random_seed, pytorch_seed: pytorch_seed, numpy_seed: numpy_seed,
  dataset_reader: { type: 'paired_dataset_reader', scores: use_scores, lazy: false },
  validation_dataset_reader: {
    type: 'reranking_dataset_reader',
    scores: use_scores,
    lazy: true
  },
  evaluate_on_test: true,
  train_data_path: Pathify(dataset + 'train.json'),
  validation_data_path: Pathify(dataset + 'validation_' + lang + '.json'),
  test_data_path: Pathify(dataset + 'test_' + lang + '.json'),
  model: {
    type: 'letor_training',
    dropout: std.extVar('dropout'),

    doc_field_embedder: Embedder(std.extVar('doc_embeddings'), embedding_dims, projection=doc_projection),
    doc_encoder: doc_encoder,

    query_field_embedder: Embedder(std.extVar('query_embeddings'), embedding_dims),
    query_encoder: query_encoder,

    use_batch_norm: std.extVar('use_batch_norm'),
    use_attention: std.extVar('use_attention'),
    ranking_loss: std.extVar('ranking_loss'),

    scorer: Scorer(embedding_dims, use_scores),
    total_scorer: {
      input_dim: 2,
      num_layers: 1,
      hidden_dims: [1],
      activations: ['linear'],
      dropout: [0.0]
    },

    validation_metrics: {
      map: {
        type: 'map',
        corrections_file: Pathify(dataset + 'validation_' + lang + '_scoring.json'),
        k: 1000
      },
      test_map: {
        type: 'map',
        corrections_file: Pathify(dataset + 'test_' + lang + '_scoring.json'),
      }
    }
  } + if std.extVar('use_idfs') then { idf_embedder: Embedder(idf_weights, 1) } else {},
  iterator: {
    type: 'bucket',
    sorting_keys: [['docs', 'list_num_tokens']],
    batch_size: std.extVar('batch_size')
  },
  validation_iterator: {
    type: 'bucket',
    sorting_keys: [['docs', 'num_fields']],
    batch_size: 2
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    cuda_device: 0,
    grad_clipping: std.extVar('clipping'),
    validation_metric: '+map',
    optimizer: {
      type: 'adam',
      lr: lr,
      weight_decay: l2
    }
  }
}
