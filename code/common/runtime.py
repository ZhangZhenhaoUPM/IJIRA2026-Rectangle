"""Build native graph kernels and materialize resolved protocols per process."""
from pathlib import Path
import atexit
import json
import shutil
import subprocess
import tempfile
from common.configuration import resolve_shared

GRAPH_ROOT = None


def prepare_graph(root, protocols, geometry):
    global GRAPH_ROOT
    GRAPH_ROOT = Path(tempfile.mkdtemp(prefix='ridge_'))
    atexit.register(shutil.rmtree, GRAPH_ROOT, ignore_errors=True)
    shutil.copytree(Path(root) / 'code/native/graph', GRAPH_ROOT / 'lib')
    for name, protocol in resolve_shared(protocols, geometry).items():
        path = GRAPH_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol), encoding='utf-8')
    for source in sorted((GRAPH_ROOT / 'lib').rglob('*.cpp')):
        subprocess.run(['g++', '-O2', '-shared', '-std=c++17', '-fPIC',
                        '-ffp-contract=off', str(source), '-o', str(source.with_suffix('.so'))],
                       check=True, capture_output=True)
    return GRAPH_ROOT
