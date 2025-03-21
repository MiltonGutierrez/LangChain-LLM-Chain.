# Build a Simple LLM Application with Chat Models and Prompt Templates

In this quickstart, we'll show you how to build a simple LLM application with LangChain. This application will translate text from English into another language. This is a relatively simple LLM application - it's just a single LLM call plus some prompting. Still, this is a great way to get started with LangChain - a lot of features can be built with just some prompting and an LLM call!

## After reading this tutorial, you'll have a high-level overview of:

- Using language models
- Using prompt templates
- Debugging and tracing your application using LangSmith

Let's dive in!

## Setup

### Jupyter Notebook

This and other tutorials are perhaps most conveniently run in a Jupyter notebook. Going through guides in an interactive environment is a great way to better understand them. See [here](#) for instructions on how to install.

### Installation

To install LangChain, run:

```bash
pip install langchain
conda install langchain -c conda-forge
```

For more details, see our [Installation guide](#).

## LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with LangSmith.

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## Using Language Models

First up, let's learn how to use a language model by itself. LangChain supports many different language models that you can use interchangeably. For details on getting started with a specific model, refer to [supported integrations](#).

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

Let's first use the model directly. ChatModels are instances of LangChain Runnables, which means they expose a standard interface for interacting with them. To simply call the model, we can pass in a list of messages to the `.invoke` method.

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

model.invoke(messages)
```

Output:
```python
AIMessage(content='Ciao!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 20, 'total_tokens': 23}})
```

:::tip
If we've enabled LangSmith, we can see that this run is logged to LangSmith, and we can see the LangSmith trace. The LangSmith trace reports token usage information, latency, standard model parameters (such as temperature), and other information.
:::

Note that ChatModels receive message objects as input and generate message objects as output. In addition to text content, message objects convey conversational roles and hold important data, such as tool calls and token usage counts.

LangChain also supports chat model inputs via strings or OpenAI format. The following are equivalent:

```python
model.invoke("Hello")

model.invoke([{ "role": "user", "content": "Hello" }])

model.invoke([HumanMessage("Hello")])
```

### Streaming

Because chat models are Runnables, they expose a standard interface that includes async and streaming modes of invocation. This allows us to stream individual tokens from a chat model:

```python
for token in model.stream(messages):
    print(token.content, end="|")
```

Output:
```plaintext
|C|iao|!||
```

You can find more details on streaming chat model outputs in [this guide](#).

## Prompt Templates

Right now we are passing a list of messages directly into the language model. Where does this list of messages come from? Usually, it is constructed from a combination of user input and application logic. This application logic usually takes the raw user input and transforms it into a list of messages ready to pass to the language model. Common transformations include adding a system message or formatting a template with the user input.

Prompt templates are a concept in LangChain designed to assist with this transformation. They take in raw user input and return data (a prompt) that is ready to pass into a language model.

Let's create a prompt template here. It will take in two user variables:

- `language`: The language to translate text into
- `text`: The text to translate

```python
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
```

Note that `ChatPromptTemplate` supports multiple message roles in a single template. We format the `language` parameter into the system message, and the user text into a user message.

The input to this prompt template is a dictionary. We can play around with this prompt template by itself to see what it does:

```python
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)
```

Output:
```python
ChatPromptValue(messages=[SystemMessage(content='Translate the following from English into Italian'), HumanMessage(content='hi!')])
```

We can see that it returns a `ChatPromptValue` that consists of two messages. If we want to access the messages directly:

```python
prompt.to_messages()
```

Output:
```python
[SystemMessage(content='Translate the following from English into Italian'), HumanMessage(content='hi!')]
```

Finally, we can invoke the chat model on the formatted prompt:

```python
response = model.invoke(prompt)
print(response.content)
```

Output:
```plaintext
Ciao!
```



