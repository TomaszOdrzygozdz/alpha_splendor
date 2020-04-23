from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import argparse
import sys
import gin
import tkinter
import webbrowser
from tkinter import messagebox

from alpacka import metric_logging
from alpacka.runner import Runner
from alpacka.system_paths import EXTRA_SYSTEM_PATHS
from alpacka.utils.neptune_logger_generator import NeptuneAPITokenException, configure_neptune, Experiment


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory.')
    parser.add_argument(
        '--config_file', action='append',
        help='Gin config files.'
    )
    parser.add_argument(
        '--project_root_path', action='append',
        help='Project root path'
    )
    parser.add_argument(
        '--config', action='append',
        help='Gin config overrides.'
    )
    parser.add_argument(
        '--neptune', action='store_true',
        help='Adds logging metrics to Neptune'
    )
    parser.add_argument(
        '--extra_paths', action='store_true',
        help='Adding extra paths to sys.paths'
    )
    parser.add_argument(
        '--additional_import', action='store_true',
        help='Importing from outside Alpaca'
    )
    parser.add_argument(
        '--open_link', action='store_true',
        help='Open Neptune experiment in web browser'
    )
    return parser.parse_args()

def _ask_for_neptune():
    root = tkinter.Tk()
    root.withdraw()
    use_neptune = messagebox.askquestion("Alpaca runner", "Do you want to log to Neptune?")
    return use_neptune == 'yes'

def _show_neptune_link(neptune_link):
    print(f'========================= Neptune link ========================= \n   {neptune_link}\n=============='
          f'=========== Neptune link ========================= ')

def _add_extra_sys_paths():
    for extra_path in EXTRA_SYSTEM_PATHS:
        sys.path.append(extra_path)

def _additional_import():
    import alpacka.additional_imports

if __name__ == '__main__':
    args = _parse_args()

    if args.extra_paths:
        print('Adding extra system paths.')
        _add_extra_sys_paths()
    if args.additional_import:
        print('Importing from outside Alpaca.')
        _additional_import()

    gin.parse_config_files_and_bindings(args.config_file, None)
    use_neptune = _ask_for_neptune()
    if use_neptune:
        experiment = Experiment()
        experiment.parse_params_from_gin_config(args.config_file)
        neptune_logger, neptune_link = configure_neptune(experiment=experiment)
        metric_logging.register_logger(neptune_logger)
        _show_neptune_link(neptune_link)
        if args.open_link:
            webbrowser.open(neptune_link)

    runner = Runner(output_dir=args.output_dir)
    runner.run()