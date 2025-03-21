
### Escuela Colombiana de Ingeniería

### Arquitectura Empresarial - AREP

# Build a Simple LLM Application with Chat Models and Prompt Templates

This tutorial introduces building a simple LLM application with LangChain to translate text from English into another language. The application relies on a single LLM call combined with prompt templating. Although it's a basic example, it demonstrates how many features can be built with just prompting and an LLM call, making it a great starting point for working with LangChain.

## After reading this tutorial, you'll have a high-level overview of:

- Using language models
- Using prompt templates
- Debugging and tracing your application using LangSmith


## Setup
### Prerequisites 
- Python installed (version > 3).
- Access to a terminal.
- Basic python programming knowledge. 

### Jupyter Notebook

#### Installation

```bash
pip install jupyterlab
pip install notebook
```

#### Create a notebook.
```bash
jupyter notebook
```
or
```bash
python -m notebook
```

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
![image](https://github.com/user-attachments/assets/83642d1f-e00d-498a-bd1c-bb76e1ba1cf9)


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
![image](https://github.com/user-attachments/assets/b3f8a9af-02f4-425e-ad47-9d3e375e117a)

```

## References:
[LangChain](https://python.langchain.com/docs/tutorials/llm_chain/)
