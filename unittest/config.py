import pytest
import sys
sys.path.insert(0, "/home/jzhu4/projects/work/harmony-curr/harmony-service-lib-py")
import harmony
from transform import HarmonyAdapter
from harmony.message import Message
from argparse import ArgumentParser

"""
@pytest.fixture
def adapter():
    def _method(message):
        return HarmonyAdapter(Message(message))
    return _method
"""

#define a class adapter

class test_adapter(HarmonyAdapter):
    def __init__(self, messagestr):
        self.adapter = HarmonyAdapter(Message(messagestr))
    
