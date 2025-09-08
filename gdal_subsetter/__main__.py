"""
=========
__main__.py
=========

Runs the harmony_gdal CLI
"""
import os
from argparse import ArgumentParser

from harmony import is_harmony_cli, run_cli, setup_cli

from .transform import HarmonyAdapter


def main():
    """Parses command line arguments and invokes the appropriate method to respond to them

    Returns
    -------
    None
    """

    parser = ArgumentParser(
        prog='harmony-gdal-adapter', description='Run the GDAL service',
    )

    setup_cli(parser)
    harmony_args = parser.parse_args()

    if is_harmony_cli(harmony_args):
        run_cli(parser, harmony_args, HarmonyAdapter)
    else:
        parser.error("Only --harmony CLIs are supported")


if __name__ == "__main__":
    #os.environ["FALLBACK_AUTHN_ENABLED"] = 'true'
    os.environ["BUFFER"] = '{"degree":0.0001, "meter":10.0}'

    main()
