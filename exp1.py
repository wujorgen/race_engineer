# dont change this. this is a working demo from chainlit.
# this does not have vectorized knowledge retreival or conversation history yet.

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    # YOU CAN EITHER USE THIS CODE BLOCK
    # text = await runnable.ainvoke({"question": message.content},callbacks=[cl.AsyncLangchainCallbackHandler()])
    # msg = cl.Message(content=text)

    # OR THIS BLOCK TO MAKE THE TEXT GO TO THE SCREEN ALL PRETTY
    # notice a chainlist message is used.
    # the astream (actually ainvoke) call to runnable is chunked
    # i dont think the order of the chunks is messed up, its just async python requiring everything contained to be async compatible.
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
