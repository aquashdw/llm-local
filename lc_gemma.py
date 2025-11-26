from langchain_community.chat_models import ChatLlamaCpp

from msg_factory import ChatMessage, ChatMessageContent
from utils import flush_buffer, forget_messages

N_CTX = 1024
N_MAX_TOKENS = 128


def load_agent(n_ctx: int, n_max_tokens: int) -> ChatLlamaCpp:
    return ChatLlamaCpp(
        model_path='gemma-3-4b-it-q4_0.gguf',
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        max_tokens=n_max_tokens,
        verbose=False,
    )


llm = load_agent(N_CTX, N_MAX_TOKENS)

messages = [
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

output_stream = llm.stream(
    input=messages,
)

full_output = []
buffer = []

for chunk in output_stream:
    if chunk.content:
        buffer.append(chunk.content)
    if sum(map(len, buffer)) > 90:
        full_output.append(flush_buffer(buffer))
if buffer:
    full_output.append(flush_buffer(buffer))

messages.append(ChatMessage('assistant', ''.join(full_output)))






while True:
    try:
        prompt = input('prompt: ')
        if prompt.startswith('.exit'):
            break
        message = ChatMessage(
            'user',
            content=[
                ChatMessageContent('text', text=prompt, )
            ]
        )
        messages.append(message)
        output_stream = llm.stream(
            input=messages,
        )

        full_output = []
        buffer = []

        for chunk in output_stream:
            if chunk.content:
                buffer.append(chunk.content)
            if sum(map(len, buffer)) > 90:
                full_output.append(flush_buffer(buffer))
        if buffer:
            full_output.append(flush_buffer(buffer))

        messages.append(ChatMessage('assistant', ''.join(full_output)))
        messages = forget_messages(llm.get_num_tokens, N_CTX - N_MAX_TOKENS * 2, messages)
    except KeyboardInterrupt:
        break


