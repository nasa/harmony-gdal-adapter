"""
=========
__main__.py
=========

Runs the harmony_gdal CLI
"""
import argparse
import logging
import time
import os

import harmony
from .transform import HarmonyAdapter


def main():
    """
    Parses command line arguments and invokes the appropriate method to respond to them

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(
        prog='harmony-gdal', description='Run the GDAL service'
    )

    harmony.setup_cli(parser)
    args = parser.parse_args()

    if (harmony.is_harmony_cli(args)):
        harmony.run_cli(parser, args, HarmonyAdapter)
    else:
        parser.error("Only --harmony CLIs are supported")


if __name__ == "__main__":
    #os.environ["FALLBACK_AUTHN_ENABLED"] = 'true'
    os.environ["BUFFER"] = '{"degree":0.01, "meter":10.0}'

    main()
