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
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id='google/gemma-3-12b-it-qat-q4_0-gguf',
	filename='gemma-3-12b-it-q4_0.gguf',
    hf_token=os.getenv('HF_TOKEN'),
    n_gpu_layers=-1,
    verbose=False,
)

# output_stream = llm.create_chat_completion(
#     messages = [
#         {
#             'role': 'user',
#             'content': [
#                 {
#                     'type': 'text',
#                     'text': 'Have you seen Matrix 2: Reloaded? What is the meaning of the conversation the Architect and Neo had at the near end of the movie? within 3 sentences.'
#                 },
#             ]
#         }
#     ],
#     stream=True,
# )

# for chunk in output_stream:
#     delta = chunk['choices'][0]['delta']
#     if delta:
#         print(delta.get('content', f'{delta}\n'), end='')
#         if delta.get('content') == '.':
#             print()

output_blob = llm.create_chat_completion(
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Have you seen Matrix 2: Reloaded? What is the meaning of the conversation the Architect and Neo had at the near end of the movie? within 3 sentences.'
                },
            ]
        }
    ],
)

from pprint import pprint
pprint(output_blob.get('choices')[0].get('message').get('content'))
