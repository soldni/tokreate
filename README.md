<h1 align="center">Tokreate</h1>

<p align="center">
    <img alt="Tokreate official logo" src="docs/res/tokreation_1x.png" width="50%">
</p>

A library to create tokens using LLMs.


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
```
