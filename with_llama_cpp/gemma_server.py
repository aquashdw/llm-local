from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Optional

import uuid
import grpc

from gemma import load_llm
from chatting_pb2 import CompletionRequest, Role, CompletionResponseChunk, StartChatResponseChunk
from chatting_pb2_grpc import ChatServiceServicer, add_ChatServiceServicer_to_server
from msg_factory import ChatMessage, ChatMessageContent
from utils import forget_messages


class ChatService(ChatServiceServicer):
    def __init__(self, n_ctx: int, system: Optional[str], ):
        """
        :param n_ctx: context window size
        :param system: system prompt for each session
        """
        self.llm = load_llm(n_ctx)
        self.messages = {}
        self.system = None
        if system:
            self.system = ChatMessage('system', system)

        self.n_max_tokens = n_ctx // 8
        self.memory_limit = n_ctx - self.n_max_tokens * 2

    def StartChat(self, request, context):
        session_id = str(uuid.uuid4())
        self.messages[session_id] = [self.system, ]
        messages = self.messages[session_id]
        idx = 0
        for content in self.yield_content(
                messages,
                ChatMessage('user', '')
        ):
            idx += 1
            yield StartChatResponseChunk(
                id=session_id,
                idx=idx,
                role=Role.ASSISTANT,
                content=content,
            )

    def RequestChat(self, request: CompletionRequest, context):
        if request.id not in self.messages:
            raise grpc.RpcError('id not found')
        prompt = request.prompt
        idx = 0
        for content in self.yield_content(
                self.messages.get(request.id),
                ChatMessage('user', content=[
                    ChatMessageContent(content_type='text', text=prompt)
        ])):
            idx += 1
            yield CompletionResponseChunk(
                idx=idx,
                role=Role.ASSISTANT,
                content=content,
            )

    def yield_content(self, messages, message):
        messages.append(message)
        output_stream = self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=self.n_max_tokens,
        )
        full_output = []
        for block in output_stream:
            delta = block.get('choices')[0].get('delta')
            if delta.get('content'):
                content = delta.get('content')
                full_output.append(content)
                yield content
            # leave for debugging
            else:
                print('found block not content')
                pprint(block)
        messages.append(ChatMessage('assistant', ''.join(full_output)))
        forget_messages(self.llm.tokenizer(), self.memory_limit, messages)


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
