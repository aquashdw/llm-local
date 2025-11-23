from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Optional

import grpc

from gemma import load_llm
from chatting_pb2 import CompletionRequest, Role, CompletionResponseChunk
from chatting_pb2_grpc import ChatServiceServicer, add_ChatServiceServicer_to_server
from msg_factory import ChatMessage, ChatMessageContent
from utils import forget_messages


class ChatService(ChatServiceServicer):
    def __init__(self, n_ctx: int, system: Optional[str], ):
        self.llm = load_llm(n_ctx)
        self.messages = []
        if system:
            self.messages.append(ChatMessage(
                    'system',
                    system
                ))
        self.tokens_per_gen = n_ctx // 8
        self.memory_limit = n_ctx - self.tokens_per_gen * 2

    def RequestChat(self, request: CompletionRequest, context):
        prompt = request.prompt
        self.messages.append(ChatMessage(
            'user',
            content=[
                ChatMessageContent('text', text=prompt,)
            ]
        ))
        output_stream = self.llm.create_chat_completion(
            messages=self.messages,
            stream=True,
            max_tokens=self.tokens_per_gen,
        )
        idx = 0
        full_output = []
        for block in output_stream:
            delta = block.get('choices')[0].get('delta')
            if delta.get('content'):
                idx += 1
                content = delta.get('content')
                full_output.append(content)
                yield CompletionResponseChunk(
                    idx=idx,
                    role=Role.ASSISTANT,
                    content=content,
                )
            # to check future usage
            else:
                print('found block not content')
                pprint(block)

        self.messages.append(ChatMessage('assistant', ''.join(full_output)))
        forget_messages(self.llm.tokenizer(), self.memory_limit, self.messages,)


if __name__ == '__main__':
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    add_ChatServiceServicer_to_server(
        ChatService(
            n_ctx=1024,
            system="""
            you are a general assistant. 
            you are assisting a person who wants to know essentials, and details matter only when necessary.
            So unless requested, keep your answers around 3 ~ 5 sentences, 
            pointing out the most significant information.
            start the chat session with a greeting and introduction of yourself.
            """,
        ),
        server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print('server started, calling wait_for_termination')
    server.wait_for_termination()
