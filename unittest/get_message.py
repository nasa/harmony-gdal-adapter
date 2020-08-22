import argparse
import harmony
import harmony
from transform import HarmonyAdapter
import sys
import logging
from harmony.message import Message

from harmony.util import CanceledException, receive_messages, delete_message, change_message_visibility, setup_stdout_log_formatting

#from .transform import HarmonyAdapter

def get_message():

    parser = argparse.ArgumentParser(
        prog='harmony-gdal', description='Run the GDAL service')

    harmony.setup_cli(parser)

    parser.add_argument('--message-file', required=True)


    args = parser.parse_args()

    if (harmony.is_harmony_cli(args)):
         if args.harmony_action == 'start':
            start(HarmonyAdapter, args.harmony_queue_url, args.harmony_visibility_timeout, args.message_file)
    else:
        parser.error("Only --harmony CLIs are supported")

def start(AdapterClass, queue_url, visibility_timeout_s, message_file):
    """
    Handles --harmony-action=start by listening to the given queue_url and invoking the
    AdapterClass on any received messages

    Parameters
    ----------
    AdapterClass : class
    The BaseHarmonyAdapter subclass to use to handle service invocations
    queue_url : string
    The SQS queue to listen on
    """
    for receipt, message in receive_messages(queue_url, visibility_timeout_s):
        
        with open(message_file,'w') as outfile:
            outfile.write(message)
        return
           



if __name__ == "__main__":

    get_message()

    print("got the message....")

