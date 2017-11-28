#!/usr/bin/env python
from __future__ import print_function

import mido

class CrowdAIMidi:
    def __init__(self, fileName):
        self.fileName = fileName
        self.init_midi()

    def init_midi(self):
        self.midifile = mido.MidiFile(self.fileName)
        print(self.midifile)
