from __future__ import print_function
import crowdai
import argparse
import mido

parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--api_key', dest='api_key', action='store', required=True)
parser.add_argument('--midi_file', dest='midi_file', action='store', required=True)
args = parser.parse_args()

print(args)

midifile = mido.MidiFile(args.midi_file)
assert midifile.length > 3600 - 10 and midifile.length < 3600 + 10
assert len(midifile.tracks) == 1

challenge = crowdai.Challenge("AIGeneratedMusicChallenge", args.api_key)
challenge.submit(args.midi_file)
