from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.agents import load_tools, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import AIPluginTool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
os.environ['OPENAI_API_KEY'] = open('baselines/openai_key.txt').read()
os.environ['GOOGLE_API_KEY'] = open('baselines/gs_key.txt').read().split('\n')[1]
os.environ['GOOGLE_CSE_ID'] = open('baselines/gs_key.txt').read().split('\n')[0]

if __name__ == '__main__':
    # search = GoogleSearchAPIWrapper(k=1)
    # tools = [
    #     Tool(
    #         name = 'search',
    #         func=search.run,
    #         description=""
    #     )
    # ]
    tool = AIPluginTool.from_plugin_url("https://chrome.google.com/webstore/detail/webchatgpt-chatgpt-with-i/lpfemeioodjbpieminkklglpmhlngfcn?utm_source=chrome-ntp-icon")
    tools = load_tools(["requests_all"])
    tools += [tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description",
                                   verbose=True, memory=memory)
    agent_chain.run(input="when was OpenAI established?")