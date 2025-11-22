from dotenv import load_dotenv

from msg_factory import ChatMessage, ChatMessageContent
from utils import flush_buffer, forget_messages

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
        messages = forget_messages(llm.tokenizer(), N_CTX - N_MAX_TOKENS * 2, messages)


    except KeyboardInterrupt:
        break

