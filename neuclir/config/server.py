from datetime import datetime

from flask import Flask, abort, flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'some_really_long_random_string_here'

# local embedding_dims = std.extVar('embedding_dims');
# local dataset = std.extVar('dataset');
# local lang = std.extVar('lang');
# local idf_weights = std.extVar('idf_weights');
# local dan = std.extVar('dan');
# local doc_projection = std.extVar('doc_projection');
# local averaged = std.extVar('averaged');
# local num_filters = std.extVar('num_filters');
# local query_averaged = std.extVar('query_averaged');
# local l2 = std.extVar('l2');
# local lr = std.extVar('lr');

variables = {
    'hyperparameters': [],
    'architecture': [
        {'name': 'embedding_dims', 'title': 'Embedding Dimensions', 'type': 'text'},
        {'name': 'idf_weights', 'title': 'Use IDF Weights', 'type': 'bool'},
        {'name': 'dan', 'title': 'Use Averaging Composer', 'type': 'bool'}
    ],
    'dataset': []
}

def render_jsonnet():
    pass

@app.route('/')
def configure():
    return render_template('configure.html')

if __name__ == '__main__':
    app.run()
