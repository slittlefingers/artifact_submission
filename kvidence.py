"""KVidence: acquire memory, extract cache tensors, and reconstruct content."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Use COMMAND --help for its options. Run from the artifact root.')
    parser.add_argument('command', choices=['acquire', 'extract', 'symbols', 'reconstruct', 'experiment'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    rest = argv[1:]
    routes = {
        'acquire': ('acquisition/acquire.py', []),
        'extract': ('extraction/run.py', ['extract']),
        'symbols': ('extraction/run.py', ['symbols']),
        'reconstruct': ('reconstruction/run.py', []),
    }
    if command == 'experiment':
        from experiments.run_all import SECTIONS
        experiments = {'5.1': 'experiments/5.1 Experimental Setup/run.py',
                       **{key: 'experiments/'+folder+'/run.py' for key, folder, _ in SECTIONS},
                       'all': 'experiments/run_all.py'}
        experiment_parser = argparse.ArgumentParser(prog='kvidence.py experiment')
        experiment_parser.add_argument('section', choices=experiments)
        if not rest or rest[0] in ('-h', '--help'):
            experiment_parser.print_help()
            return 0
        section = experiment_parser.parse_args(rest[:1]).section
        script, prefix, rest = experiments[section], [], rest[1:]
    else:
        script, prefix = routes[command]
    return subprocess.run([sys.executable, '-B', str(ROOT/script), *prefix, *rest]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
