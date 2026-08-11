"""Checks the index-space assumption test.py relies on when writing .pl files.

map/test           == [IO ..., macro ...]      -> indexed by the full pos tensor
map/test_clustered == [macro ..., cluster ...] -> indexed by pos[num_io:]

If these ever drift apart, write_bench_files() silently emits placements with
the wrong coordinates instead of failing, so assert the shapes here.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGNS = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4',
           'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4']


def load(subdir, design):
    with open(os.path.join(ROOT, 'map', subdir, f'{design}.json')) as f:
        return json.load(f)


def test_map_index_spaces():
    for design in DESIGNS:
        plain = load('test', design)
        clustered = load('test_clustered', design)
        macros = [n for n in clustered if not n.startswith('cluster_')]

        assert macros, f'{design}: clustered map has no macro entries'
        assert clustered[:len(macros)] == macros, \
            f'{design}: clustered map must list macros before clusters'
        assert plain[-len(macros):] == macros, \
            f'{design}: plain map must list macros last, after the IOs'
        assert len(plain) > len(macros), f'{design}: plain map is missing IO entries'


if __name__ == '__main__':
    test_map_index_spaces()
    print(f'ok: map index spaces consistent for {len(DESIGNS)} designs')
