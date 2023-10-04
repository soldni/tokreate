<h1 align="center">Tokreate</h1>

<p align="center">
    <img alt="Tokreate official logo" src="https://github.com/soldni/tokreate/blob/main/assets/img/tokreate@1x.png?raw=true" width="45%">
</p>

A minimal library to create tokens using LLMs.

**I refuse to write more than 200 lines of code for this.**
No [LangChain](https://www.langchain.com) shenenigans.
See [tokereate.py](https://github.com/soldni/tokreate/blob/main/src/tokreate/tokereate.py) to see where we stand.


Tokreate is [available on PyPI](https://pypi.org/project/tokreate/)! Simply run:

```shell
pip install tokreate
```

to get started.

## Usage

Start by setting up a call action:


<!-- {% raw %} -->
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
<!-- {% endraw %} -->

Actions can be chained together:


<!-- {% raw %} -->
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

history = (greetings >> tell_joke).run(name="Alice", animal="snake")

for turn in history:
    print(turn)

# Output:
# user>>> Hello, my name is Alice; what's yours?
# assistant>>> Hello Alice, nice to meet you! I'm ChatGPT. How can I assist you today?
# user>>> Can you finish this joke: What do you call a snake with no legs?
# assistant>>> What do you call a snake with no legs? A "legless" reptile!
```
<!-- {% endraw %} -->

You can use parsers to extract information from any call:

<!-- {% raw %} -->
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
*_, last_turn = (random_numbers >> json_parser).run(random_count='five')

parsed_content = last_turn.state["json.loads"]
print(f"Response {parsed_content} (type: {type(parsed_content)})")

# Output:
# Response {'numbers': [4, 9, 2, 7, 1]} (type: <class 'dict'>)
```
<!-- {% endraw %} -->

You can switch to different models at any point:

<!-- {% raw %} -->
```python
from tokreate import CallAction

greetings = CallAction(
    prompt="Hello, my name is {{ name }}; what's yours?",
    model="claude-instant-v1.1"
)

history = greetings.run(name="Alice")

for turn in history:
    print(turn)

# Output:
# user>>> Hello, my name is Alice; what's yours?
# assistant>>> I'm Claude, an AI assistant created by Anthropic.
```
<!-- {% endraw %} -->

Let's try with one of the TogetherAI models:

<!-- {% raw %} -->
```python
from tokreate import CallAction

greetings = CallAction(
    prompt="Hello, my name is {{ name }}; what's yours?",
    model="togethercomputer/llama-2-70b-chat",
    history=False   # TogetherAI models don't support history at API level yet :( you can make your own in prompt
)

history = greetings.run(name="Alice")

for turn in history:
    print(turn)

# Output:
# user>>> Hello, my name is Alice; what's yours?
# assistant>>>  Hello Alice! My name is ChatBot, it's nice to meet you. How can I assist you today? Is there a specific topic you'd like to discuss or ask me a question about?
```
<!-- {% endraw %} -->
