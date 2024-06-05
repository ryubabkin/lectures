# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 0: Preparation
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")
AZURE_OPENAI_URL = "https://testgptzen.openai.azure.com"
AZURE_OPENAI_MODEL = "gpt-4"
AZURE_DEPLOYMENT = "gpt-4"
AZURE_OPENAI_VERSION = "2023-05-15"
AZURE_OPENAI_KEY = ". . ."

# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 1: LLM initialization
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
from langchain_community.chat_models import AzureChatOpenAI

llm = AzureChatOpenAI(
    openai_api_type="azure",
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_URL,
    deployment_name=AZURE_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_VERSION,
    model=AZURE_OPENAI_MODEL,
    callbacks=[],  # Will be used in agents part
    streaming=True,  # Streaming mode - response will be streamed token-by-token
    temperature=0.5,  # Temperature of LLM response generation
    max_tokens=1024  # Output tokens limit
)
result = llm.predict("Hello, world!")
print(result)
print(f'IN tokens: {len(encoder.encode("Hello, world!"))}, OUT tokens: {len(encoder.encode(result))}')

# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 2: Prompt Engineering
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
   template="Give responses to the following question: {question}. Be brief and give easy to understand responses. Your response should be in {language}",
   input_variables=["question", "language"]
)
prompt_formatted_str: str = prompt.format(
   question="Who is Santa Claus?",
   language="Spanish"
)
prediction = llm.predict(prompt_formatted_str)

# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 3: Chains
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

from langchain.chains import LLMChain, LLMMathChain, SimpleSequentialChain

prompt = PromptTemplate(
   template="Generate a simple math task for adding {entity}.",
   input_variables=["entity"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
math_chain = LLMMathChain(llm=llm, verbose=True)
combined_chain = SimpleSequentialChain(chains=[llm_chain, math_chain])

print(combined_chain.run("apples"))

# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 4: Agents and Tools
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

from langchain.agents import Tool, load_tools
from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)
tools_set = [
    Tool(
        name='Language Model',
        func=LLMChain(llm=llm, prompt=prompt).run,
        description='Use this tool for general purpose queries and logic'
    ),
    Tool(
        name='Symbolic Calculator',
        func=LLMSymbolicMathChain.from_llm(llm).run,
        description='Useful for when you need to answer questions about math involving symbolic calculations and algebra.'
    ),
    load_tools(["llm-math"], llm=llm)[0]
]


from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools_set,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.run("A 100 meter long train passes through a tunnel that is 100 meter long at a speed of 100 km/h. During how long time is the entire train inside the tunnel?")

# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 5: Conversation Memory, Chat
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

conversation_with_memory = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=False
)
print(conversation_with_memory.predict(input="Hi!"))
print(conversation_with_memory.predict(input="What was my previous message?"))


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 6: Vector Stores and Similarity Search
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

import os
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] ='. . .'

raw_documents = TextLoader('/home/user/story.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

query = 'Who said "No, I dare not take it."?'
docs = db.similarity_search(query)
print(docs[0].page_content)


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
"""
Example 7: Chat Bot
"""
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.agents import ZeroShotAgent, AgentExecutor

prefix = "You are an AI assistant developed by zenseact to help people understand GenAI."

suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}
"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools_set,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

chat_history = ChatMessageHistory(messages=[])

window_memory = ConversationSummaryBufferMemory(
    llm=llm,
    chat_memory=chat_history,
    input_key="input",
    memory_key="chat_history"
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=window_memory
)

agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=tools_set,
    verbose=True,
    handle_parsing_errors=True,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools_set,
    memory=window_memory,
    verbose=True,
    handle_parsing_errors=True
)

print(agent_executor.run(input="Hello"))
print(agent_executor.run(input="How are you?"))
print(agent_executor.run(input="what is 2^3 + 5?"))
print(agent_executor.run(input="What was my first question?"))