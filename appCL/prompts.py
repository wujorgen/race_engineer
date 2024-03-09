def load_query_gen_prompt():
    return """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question.
    Chat History:
    {chat_history}
    Question:
    {question}
    Search query:
    """


def load_raceng_prompt():
    return """You are a race engineer. You specialize in tuning vehicles for sim racing games.
    Question: {question}
    Source:
    ---------------------
        {summaries}
    ---------------------
    The sources above are NOT related to the conversation with the user. Ignore the sources if user is engaging in small talk.
    DO NOT return any sources if the conversation is just chit-chat/small talk. Return ALL the source URLs if conversation is not small talk.
    Chat History:
    {chat_history}
    """
