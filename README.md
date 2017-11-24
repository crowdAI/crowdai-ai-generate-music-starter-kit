![CrowdAI-Logo](https://github.com/crowdAI/crowdai/raw/master/app/assets/images/misc/crowdai-logo-smile.svg?sanitize=true)

# crowdai-ai-generate-music-starter-kit

Starter kit for the [AI Generated Music Challenge](https://www.crowdai.org/challenges/ai-generated-music-challenge) on [CrowdAI](https://www.crowdai.org/).

**Coming Soon**
A getting started guide to generating music.

# Installation
```
pip install -U crowdai
pip install -U mido
```
**NOTE** : This challenge requires the crowdai client with version `>= 1.0.12`.

# Usage

```
import crowdai
import mido

midi_file_path="<your_midi_file_path>"
API_KEY="<your_crowdai_api_key_here>"

midifile = mido.MidiFile(midi_file_path)
assert midifile.length > 3600 - 10 and midifile.length < 3600 + 10
assert len(midifile.tracks) == 1

challenge = crowdai.Challenge("AIGeneratedMusicChallenge", API_KEY)
challenge.submit(midi_file_path)
"""
  Common pitfall: `challenge.submit` takes the `midi_file_path`
                    and not the `midifile` object
"""
```

# Your first submission
```
git clone https://github.com/crowdAI/crowdai-ai-generate-music-starter-kit
cd crowdai-ai-generate-music-starter-kit
pip install -r requirements.txt
python submit.py --api_key=<YOUR_CROWDAI_API_KEY> --midi_file=<PATH_TO_YOUR_MIDI_FILE>
```

# Author
S.P. Mohanty <sharada.mohanty@epfl.ch>
