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
	repo_id='google/gemma-3-4b-it-qat-q4_0-gguf',
	filename='gemma-3-4b-it-q4_0.gguf',
    hf_token=os.getenv('HF_TOKEN'),
    n_gpu_layers=-1,
    verbose=False,
)

# output_blob = llm.create_chat_completion(
#     messages = [
#         {
#             'role': 'user',
#             'content': [
#                 {
#                     'type': 'text',
#                     'text': 'Have you seen Matrix 2: Reloaded? What is the meaning of the conversation the Architect and Neo had at the near end of the movie? answer with 3 ~ 5 senctences.'
#                 },
#             ]
#         }
#     ],
# )

# from pprint import pprint
# pprint(output_blob.get('choices')[0].get('message').get('content'))


while True:
    try:
        prompt = input('prompt: ')
        if prompt.startswith('.exit'):
            break
        output_stream = llm.create_chat_completion(
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt,
                        },
                    ]
                }
            ],
            stream=True,
            max_tokens=2048,
        )
        output_words = []
        for block in output_stream:
            delta = block.get('choices')[0].get('delta')
            output_words.append(delta.get('content')) if delta.get('content') else None
            if sum(map(len, output_words)) > 90:
                print(''.join(output_words))
                output_words.clear()
        if output_words:
            print(''.join(output_words))

    except KeyboardInterrupt:
        break

