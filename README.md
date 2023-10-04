<h1 align="center">Tokreate</h1>

<p align="center">
    <img alt="Tokreate official logo" src="https://github.com/soldni/tokreate/blob/main/assets/img/tokreate@1x.png?raw=true" width="45%">
</p>

A minimal library to create tokens using LLMs.

**I refuse to write more than 200 lines of code for this.**
No [LangChain](https://www.langchain.com) shenanigans.
See [tokreate.py](https://github.com/soldni/tokreate/blob/main/src/tokreate/tokreate.py) to see how well I'm sticking to this.


Tokreate is [available on PyPI](https://pypi.org/project/tokreate/)! Simply run:

```shell
pip install tokreate
```
to get started.


Use the following shortcut to navigate to a section and learn more about tokreate:

- [**Usage**](#usage)
- [**Design**](#design)
- [**Supported APIs**](#supported-apis)
- [**GitHub**](https://github.com/soldni/tokreate)


## Usage

Below, I show some ways tokreate can be used to prompt LLMs.

### A Simple Call Action

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
    print(repr(turn))

# user>>> Hello, my name is Alice; what's yours?
# assistant>>> Hello Alice, nice to meet you! I'm ChatGPT. How can I assist you today?
```
<!-- {% endraw %} -->

### Chaining Actions


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
    print(repr(turn))

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

# Response {'numbers': [4, 9, 2, 7, 1]} (type: <class 'dict'>)
```
<!-- {% endraw %} -->

### Using Different Models

This is an example on how to use Anthropic's models:

<!-- {% raw %} -->
```python
from tokreate import CallAction

greetings = CallAction(
    prompt="Hello, my name is {{ name }}; what's yours?",
    model="claude-instant-v1.1"
)

history = greetings.run(name="Alice")

for turn in history:
    print(repr(turn))

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
    print(repr(turn))

# user>>> Hello, my name is Alice; what's yours?
# assistant>>>  Hello Alice! My name is ChatBot, it's nice to meet you. How can I assist you today? Is there a specific topic you'd like to discuss or ask me a question about?
```
<!-- {% endraw %} -->

### Async Support

You can make calls to LLMs asynchronously thanks to async support in tokreate. Note that some APIs (e.g., Anthropic) might support very few concurrent calls by default.

<!-- {% raw %} -->
```python
import asyncio
from tokreate import CallAction


async def main():

    greetings = CallAction(
        prompt="Generate a very short description of the color {{ color }}.",
        model="gpt-3.5-turbo"
    )

    futures = []
    for color in ["red", "green", "blue"]:
        futures.append(greetings.arun(color=color))

    results = await asyncio.gather(*futures)

    for (request, response) in results:
        print(f"{request.state['color']}:", response)


if __name__ == "__main__":
    asyncio.run(main())

# red: Passionate and intense, red is a vibrant hue that evokes feelings of love, power, and energy.
# green: Green is the vibrant hue of nature, symbolizing growth, freshness, and harmony.
# blue: Blue is a cool and calming color that evokes feelings of tranquility and serenity.
```
<!-- {% endraw %} -->


## Design

I tried to keep the design of tokreate as simple as I possibly could. The library is composed of two main classes: a `Turn` and `Action`.

- **Turns** are serializable objects that represent a single turn in a conversation.
- **Actions** are objects that use prompts to call an LLMs. Actions generate turns.

### Turn APIs

Under the hood, turns are `msgspec.Struct` objects. Each turn has the following fields:

- `Turn.role`: this can be either `user` or `assistant`, depending on who generated the turn.
- `Turn.content`: the output of the LLM if the turn was generated by an action. Otherwise, this is the prompt from an action.
- `Turn.state`: a dictionary that can be used to store any information. Typically this is either variables provided when calling the `run` method of an action, or the output of a `ParseAction`.
- `Turn.meta`: a dictionary that stores metadata, typically from calling an LLM api.

For example, the following code produces a turn with the following content:

```python

from tokreate import CallAction

greetings = CallAction(
    prompt="Hello, my name is {{ name }}; what's yours?",
    model="gpt-3.5-turbo"
)

user_turn, assistant_turn = greetings.run(name="Alice")

print("User turn:")
print('\trole:', user_turn.role)
print('\tcontent:', user_turn.content)
print('\tstate:', user_turn.state)
print('\tmeta:', user_turn.meta)
print('\n')
print("Assistant turn:")
print('\trole:', assistant_turn.role)
print('\tcontent:', assistant_turn.content)
print('\tstate:', assistant_turn.state)
print('\tmeta:', assistant_turn.meta)

# User turn:
#    role: user
#    content: Hello, my name is Alice; what's yours?
#    state: {'name': 'Alice'}
#    meta: {}
# Assistant turn:
#   role: assistant
#   content: Hello Alice, nice to meet you! I'm ChatGPT. How can I assist you today?
#   state: {'name': 'Alice'}
#   meta: {'tokens_prompt': 18, 'tokens_completion': 21, 'latency': 0.47, 'messages': [{'role': 'user', 'content': "Hello, my name is Alice; what's yours?"}], 'temperature': 0, 'max_tokens': 300}
```

Note that `str(turn)` returns the `content` of the turn, and `repr(turn)` returns '{role}>>> {content}'.

### Action APIs

Actions are objects that use prompts to call an LLMs. Actions are run through the `run()` method, and always return a list of turns (i.e., a history). They can be chained together using `>>` and `<<` operators. A chain is itself an action.

The library supports two types of actions: `CallAction` and `ParseAction`. In theory, one could add actions for retrieval, planning, etc. In practice, I don't think those are in scope for this library, and can be added by users.

#### CallAction

A `CallAction` is an action that calls an LLMs. It requires two parameters:

- `prompt`: the prompt to use when calling the LLMs. Prompts are parsed using [jinja2](https://jinja.palletsprojects.com/); each prompt has access to all variables passed by the user when calling the `run()` method of the action; the `state` of the previous turn, and the `history` of the conversation so far.
- `model`: the name of the model to use. This is a string that can be found by running `python -m tokreate.registry`.

For example, the following chain of action uses many advanced features of jinja2 templating language:

<!-- {% raw %} -->
```python
from tokreate import CallAction

system = "You are a conversational assistant. You like to talk, but prefer giving short answers."

greetings_action = CallAction(
    prompt="Hello, my name is {{ name }}; what's yours?",
    model="gpt-3.5-turbo"
)
hobbies_action = CallAction(
    prompt="What are your hobbies? Mine are {% for hobby in hobbies %}{{ hobby }}{% if not loop.last %}, {% else %}.{% endif %}{% endfor %}",
    model="gpt-3.5-turbo"
)
color_action = CallAction(
    prompt="What's your favorite color? Mine is {{ favorite_color }}",
    model="gpt-3.5-turbo"
)


reflect_prompt = """
So far, you told me that:
{% for turn in history %}{% if turn.role == "assistant" %}
- "{{ turn.content.strip() }}"{% endif %}{% endfor %}

Please reflect on what you told me, and what it says about you.
""".strip()
reflect_action = CallAction(
    prompt=reflect_prompt,
    model="gpt-3.5-turbo"
)

history = (
    greetings_action
    >> hobbies_action
    >> color_action
    >> reflect_action
).run(
    name="Alice",
    hobbies=["reading", "writing", "running"],
    favorite_color="blue"
)

for turn in history:
    print(repr(turn))
```
<!-- {% endraw %} -->

### ParseAction

The parse action is used to convert the content of the last turn into a conversation using a function. The output of the conversion is saved in the state of the turn.

For example, a parse action can be used to convert the output of an LLMs into a JSON object:

<!-- {% raw %} -->
```python
import json
from tokreate import ParseAction
from tokreate.tokreate import Turn

turn = Turn(
    role="assistant",
    content='{"numbers": [4, 9, 2]}'
)

json_parser = ParseAction(
    parser=json.loads,
    name="structured_output"
)

*_, parsed_turn = json_parser.step(history=[turn])

assert parsed_turn.state["structured_output"] == {"numbers": [4, 9, 2]}
```
<!-- {% endraw %} -->

Note how `name` is used to specify the key in the state where the output of the parser should be saved. If `name` is not specified, the output of the parser is save at `{function_module}.{function_name}` (in this case, it would be at `parsed_turn.state["json.loads"]`).

## Supported APIs

This library currently supports OpenAI, Anthropic, and TogetherAI. For a list of all models, run `python -m tokreate.registry`. Please open an [issue on GitHub](https://github.com/soldni/tokreate/issues) to request support for a new API.
