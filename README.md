<h1 align="center">Tokreate</h1>

<p align="center">
    <img alt="Tokreate official logo" src="docs/res/tokreation_1x.png" width="50%">
</p>

A library to create tokens using LLMs.


## Usage

Start by writing a prompt.

```python
from tokreate import Prompt

class Greetings(Prompt):
    prompt = "Hello, my name is {{ name }}. What's yours?"

greetings = Greetings()

# alternatively...

greetings = Prompt(prompt="Hello, my name is {{ name }}. What's yours?")
```
