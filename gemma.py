from dotenv import load_dotenv

from msg_factory import ChatMessage, ChatMessageContent

load_dotenv()

"""
# transformers pipeline gemma-3-1b
from transformers import pipeline
import torch


pipe = pipeline('text-generation', model='google/gemma-3-1b-it', device='mps', torch_dtype=torch.bfloat16)

messages = [
    [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': 'You are a helpful assistant.'},]
        },
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': 'Write a poem on Hugging Face, the company'},]
        },
    ],
]

output = pipe(messages, max_new_tokens=50)
print(output)
"""

# llama_cpp
import os
from llama_cpp import Llama

N_CTX = 1024
N_MAX_TOKENS = 128

assert N_CTX - N_MAX_TOKENS * 2 > 0

llm = Llama.from_pretrained(
	repo_id='google/gemma-3-4b-it-qat-q4_0-gguf',
	filename='gemma-3-4b-it-q4_0.gguf',
    hf_token=os.getenv('HF_TOKEN'),
    n_gpu_layers=-1,
    n_ctx=N_CTX,
    verbose=False,
)


messages=[
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

output_blob = llm.create_chat_completion(messages=messages, max_tokens=1024)
content = output_blob.get('choices')[0].get('message').get('content')

from pprint import pprint
pprint(content)

messages.append(ChatMessage('assistant', content))


def flush_buffer(buffer):
    joined = ''.join(buffer)
    print(''.join(buffer))
    buffer.clear()
    return joined


def forget_messages():
    token_counts = []
    global messages
    for message in messages:
        if 'content' not in message:
            continue
        content = message.get('content')
        if isinstance(content, str):
            token_counts.append(len(llm.tokenize(message.get('content').encode('utf-8'))))
        elif isinstance(content, list):
            for content_part in content:
                if content_part.get('type') != 'text':
                    continue
                token_counts.append(len(llm.tokenize(content_part.get('text').encode('utf-8'))))
    token_total = sum(token_counts)
    token_ceil = N_CTX - N_MAX_TOKENS * 2
    if token_total <= token_ceil:
        print('forget--------')
        print(f'token count within range: {token_total}')
        print('forget--------')
        return

    new_messages = [messages[0]]
    token_removed = 0
    idx = 1
    while idx < len(token_counts) and token_total - token_removed > token_ceil:
        token_removed += token_counts[idx]
        token_removed += token_counts[idx + 1]
        idx += 2

    new_messages.extend(messages[idx:])
    messages = new_messages
    print('forget--------')
    print(f'removed total {idx - 1} messages to remove {token_removed} tokens')
    print(f'now: {token_total}')
    print('forget--------')


while True:
    try:
        prompt = input('prompt: ')
        if prompt.startswith('.exit'):
            break
        message = ChatMessage(
            'user',
            content=[
                ChatMessageContent('text', text=prompt,)
            ]
        )
        messages.append(message)
        output_stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=N_MAX_TOKENS,
        )
        full_output = []
        buffer = []
        for block in output_stream:
            delta = block.get('choices')[0].get('delta')
            if delta.get('content'):
                buffer.append(delta.get('content'))

            if sum(map(len, buffer)) > 90:
                full_output.append(flush_buffer(buffer))
        if buffer:
            full_output.append(flush_buffer(buffer))

        messages.append(ChatMessage('assistant', ''.join(full_output)))
        forget_messages()


    except KeyboardInterrupt:
        break

