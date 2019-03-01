import os
import sys
import json
import argparse

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import JsonDict


class ConvertTREC(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Convert JSON output to TREC or TSV output'''
        subparser = parser.add_parser(
                name, description=description, help='Convert JSON output to TREC or TSV.')

        subparser.add_argument('--input', type=str, help='input file', required=True)
        subparser.add_argument('--output', type=str, default='', help='output file or directory')
        subparser.add_argument('--corrections', type=str, default='', help='corrections file for empty queries')
        subparser.add_argument('--type', type=str, default='trec', help='output type [trec, tsv]')
        subparser.set_defaults(func=_convert)

        return subparser


def _convert(args: argparse.Namespace) -> None:
    output_type = 'file'
    if os.path.isdir(args.output):
        output_type = 'dir'

    for file in [args.input, args.corrections]:
        with open(file) as fp:
            for line in fp:
                query = json.loads(line)
                process_query(query, args.output, args.type, output_type)


def process_query(query: JsonDict, output: str, doctype: str = 'trec', output_type: str = 'file') -> None:
    query_id = query['query_id']

    if output_type == 'file':
        fp = open(output, 'a')
    else:
        fp = open(os.path.join(output, 'q-' + query_id + '.' + doctype), 'w')
        if doctype == 'tsv': fp.write(f'{query_id}\tquery\n')

    if 'scores' in query:
        for i, (doc_id, score) in enumerate(query['scores']):
            if doctype == 'trec':
                fp.write(f'{query_id}\tQ0\t{doc_id}\t{i+1}\t{score}\tneuclir\n')
            elif doctype == 'tsv':
                fp.write(f'{doc_id}\t{score}\n')

    fp.close()
