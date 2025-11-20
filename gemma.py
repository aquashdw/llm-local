from dotenv import load_dotenv
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
from llama_cpp import Llama, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage, \
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestAssistantMessage

llm = Llama.from_pretrained(
	repo_id='google/gemma-3-4b-it-qat-q4_0-gguf',
	filename='gemma-3-4b-it-q4_0.gguf',
    hf_token=os.getenv('HF_TOKEN'),
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)


messages=[
    ChatCompletionRequestSystemMessage(
        role='system',
        content="""
        you are a general assistant. 
        you are assisting a person who wants to know essentials, and details matter only when necessary.
        So unless requested, keep your answers around 3 ~ 5 sentences, 
        pointing out the most significant information.
        start the chat session with a greeting and introduction of yourself.
        """
    ),
    ChatCompletionRequestUserMessage(
        role='user',
        content='',
    )
]

output_blob = llm.create_chat_completion(messages=messages, max_tokens=1024)
content = output_blob.get('choices')[0].get('message').get('content')

from pprint import pprint
pprint(content)

messages.append(ChatCompletionRequestAssistantMessage(
    role='assistant',
    content=content,
))


def flush_buffer(buffer):
    joined = ''.join(buffer)
    print(''.join(buffer))
    buffer.clear()
    return joined


while True:
    try:
        prompt = input('prompt: ')
        if prompt.startswith('.exit'):
            break
        message = ChatCompletionRequestUserMessage(
            role='user',
            content=[
                ChatCompletionRequestMessageContentPartText(
                    type='text',
                    text=prompt,
                )
            ]
        )
        messages.append({
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt,
                },
            ],
        })
        output_stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=2048,
        )
        full_output = []
        buffer = []
        for block in output_stream:
            delta = block.get('choices')[0].get('delta')
            buffer.append(delta.get('content')) if delta.get('content') else None
            if sum(map(len, buffer)) > 90:
                full_output.append(flush_buffer(buffer))
        if buffer:
            full_output.append(flush_buffer(buffer))

        messages.append(ChatCompletionRequestAssistantMessage(
            role='assistant',
            content=''.join(full_output),
        ))

    except KeyboardInterrupt:
        break

