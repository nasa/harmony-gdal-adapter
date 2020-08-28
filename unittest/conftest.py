import pytest
import harmony
from transform import HarmonyAdapter
from harmony.message import Message
from argparse import ArgumentParser


@pytest.fixture

def adapter():

    def _method(message):
        return HarmonyAdapter(Message(message))

    return _method
