# crowdai-ai-generate-music-starter-kit

# Installation
```
pip install -U crowdai
pip install -U mido
```

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

# Author
S.P. Mohanty <sharada.mohanty@epfl.ch>
