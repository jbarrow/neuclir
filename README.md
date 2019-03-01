# NeuCLIR

Neural CLIR in the absence of gold labels.

## Running

There are two stages to run this:

### 1. Sampling

```
python -m data.sample
```

### 2. Training

```
jsonnet train.jsonnet > train.json
allennlp train train.json --include-package models --include-package readers -s /tmp/output
```
