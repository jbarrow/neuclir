import os
import sys
import json

from allennlp.common.util import JsonDict

def process_query(query: JsonDict, output_dir: str, doctype: str = 'trec') -> None:
    query_id = query['query_id']

    with open(os.path.join(output_dir, 'q-' + query_id + '.' + doctype), 'w') as fp:
        if doctype == 'tsv':
            fp.write(f'{query_id}\tquery\n')
        if 'scores' in query:
            for i, (doc_id, score) in enumerate(query['scores']):
                if doctype == 'trec':
                    fp.write(f'{query_id}\tQ0\t{doc_id}\t{i+1}\t{score}\tneuclir\n')
                elif doctype == 'tsv':
                    fp.write(f'{doc_id}\t{score}\n')

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage:\npython commands/json_to_trec.py [input_file].json [scoring_file].json [output_dir] [doctype]')
        sys.exit()

    _, input_file, scoring_file, output_dir, doctype = sys.argv

    with open(input_file) as fp:
        for line in fp:
            query = json.loads(line)
            process_query(query, output_dir, doctype)

    with open(scoring_file) as fp:
        for line in fp:
            query = json.loads(line)
            process_query(query, output_dir, doctype)
