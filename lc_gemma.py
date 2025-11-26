from pprint import pprint

from langchain_community.chat_models import ChatLlamaCpp

from msg_factory import ChatMessage


def load_agent(n_ctx: int) -> ChatLlamaCpp:
    return ChatLlamaCpp(
        model_path='gemma-3-4b-it-q4_0.gguf',
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        max_tokens=n_ctx // 8,
        verbose=False,
    )


llm = load_agent(1024)
output = llm.invoke(
    input=[
        ChatMessage(
            'system',
            """
            you are a general assistant. 
            you are assisting a person who wants to know essentials, and details matter only when necessary.
            So unless requested, keep your answers around 3 ~ 5 sentences, 
            pointing out the most significant information.
            start the chat session with a greeting and introduction of yourself.
            """,
        ),
        ChatMessage('user', '')
    ]
)

pprint(output.content)
