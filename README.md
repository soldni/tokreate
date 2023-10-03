<h1 align="center">Tokreate</h1>

<p align="center">
    <img alt="Tokreate official logo" src="https://github.com/soldni/tokreate/blob/main/docs/res/tokreation_1x.png?raw=true" width="50%">
</p>

A library to create tokens using LLMs.

**I refuse to write more than 200 lines of code for this.**
No [LangChain](https://www.langchain.com) shenenigans.
See [tokereate.py](https://github.com/soldni/tokreate/blob/main/src/tokreate/tokereate.py) to see where we stand.


## Usage

Start by setting up a call action:

```python
from tokreate import CallAction

greetings = CallAction(
    prompt="Hello, my name is {{ name }}; what's yours?",
    model="gpt-3.5-turbo"
)

# tokreate always returns a history of the conversation
history = greetings.run(name="Alice")

for turn in history:
    print(turn)

# Output:
# user>>> Hello, my name is Alice; what's yours?
# assistant>>> Hello Alice, nice to meet you! I'm ChatGPT. How can I assist you today?
```

Actions can be chained together:

```python

from tokreate import CallAction

greetings = CallAction(
    prompt="Hello, my name is {{ name }}; what's yours?",
    model="gpt-3.5-turbo"
)
tell_joke = CallAction(
    prompt="Can you finish this joke: What do you call a {{ animal }} with no legs?",
    model="gpt-3.5-turbo"
)

history = (greetings + tell_joke).run(name="Alice", animal="snake")

for turn in history:
    print(turn)

# Output:
# user>>> Hello, my name is Alice; what's yours?
# assistant>>> Hello Alice, nice to meet you! I'm ChatGPT. How can I assist you today?
# user>>> Can you finish this joke: What do you call a snake with no legs?
# assistant>>> What do you call a snake with no legs? A "legless" reptile!
```

You can use parsers to extract information from any call:

```python

import json
from tokreate import CallAction, ParseAction

random_numbers = CallAction(
    prompt="Return a list of {{ random_count }} random numbers.",
    system="You are an helpful assistant who responds by always returning a valid JSON object.",
    model="gpt-3.5-turbo"
)
json_parser = ParseAction(
    parser=json.loads
)
*_, last_turn = (random_numbers + json_parser).run(random_count='five')

print(f"Response {last_turn.content} (type: {type(last_turn.content)})")

# Output:
# Response {'numbers': [4, 9, 2, 7, 1]} (type: <class 'dict'>)
```
