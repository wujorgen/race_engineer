# testing the source code from
# https://betterprogramming.pub/harnessing-retrieval-augmented-generation-with-langchain-2eae65926e82

import os

import chainlit as cl
from dotenv import find_dotenv, load_dotenv
from langchain.callbacks import ContextCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

# from langchain_community.callbacks import ContextCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompts import load_query_gen_prompt, load_raceng_prompt

load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(load_query_gen_prompt())


@cl.on_chat_start
async def init():
    # define model and memory
    llm = ChatOpenAI(api_key=api_key, verbose=True, streaming=True)
    memory = ConversationTokenBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        max_token_limit=1000,
    )
    # context_callback = ContextCallbackHandler()

    # load documents, vectorize them into chroma, define doc retriever
    files_to_read = [
        "datafiles/baseline_setup.txt",
        "datafiles/coachdave_tuning_guide.txt",
    ]
    files_raw = []

    for file in files_to_read:
        with open(file, "r", encoding="utf-8") as f:
            files_raw.append(f.readlines())

    file_data = [(" ".join(f)) for f in files_raw]
    docs = [
        Document(f, metadata={"source": m}) for f, m in zip(file_data, files_to_read)
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # some prompt engineering stuff goes here that i dont really understand
    messages = [SystemMessagePromptTemplate.from_template(load_raceng_prompt())]
    # print('mem', user_session.get('memory'))
    messages.append(HumanMessagePromptTemplate.from_template("{question}"))
    prompt = ChatPromptTemplate.from_messages(messages)

    question_generator = LLMChain(
        llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True
    )

    doc_chain = load_qa_with_sources_chain(
        llm, chain_type="stuff", verbose=True, prompt=prompt
    )

    # define the retrieval chain
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        verbose=True,
        memory=memory,
        rephrase_question=False,
        # callbacks=[context_callback],
    )
    cl.user_session.set("conversation_chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("conversation_chain")

    # YOU CAN EITHER USE THIS CODE BLOCK

    text = await runnable.ainvoke(
        {"question": message.content}, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    answer = text["answer"]
    msg = cl.Message(content=answer)

    await msg.send()

    # OR THIS BLOCK TO MAKE THE TEXT GO TO THE SCREEN ALL PRETTY
    # notice a chainlist message is used.
    # the astream (actually ainvoke) call to runnable is chunked
    # i dont think the order of the chunks is messed up, its just async python requiring everything contained to be async compatible.
    # streaming the output of chains gets weird though so...we're not using it for now :(
    # see: https://github.com/langchain-ai/langchain/discussions/4444
    """
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    
    await msg.send()
    """
