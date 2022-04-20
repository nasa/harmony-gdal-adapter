import argparse
import time
import json

import harmony


def capture_op1():
    parser = argparse.ArgumentParser(
        prog='harmony-gdal', description='Run the GDAL servic')
    harmony.setup_cli(parser)
    args = parser.parse_args()

    data = args.harmony_input

    print(data)

    with open('/home/operation.json', 'w') as outfile:
        json.dump(data, outfile)

    outfile.close()

    # sleep 1 hour
    time.sleep(3600)


def capture_op2():

    data = args.harmony_input

    with open('/home/operation.json', 'w') as outfile:
        json.dump(data, outfile)

    outfile.close()

    # sleep 1 hour
    time.sleep(3600)


if __name__ == "__main__":

    capture_op1
