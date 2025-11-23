from typing import Iterable

import grpc
from chatting_pb2 import CompletionRequest, CompletionResponseChunk
from chatting_pb2_grpc import ChatServiceStub
from utils import flush_buffer

if __name__ == '__main__':
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = ChatServiceStub(channel)
        message = CompletionRequest(prompt='hello')
        output_stream: Iterable[CompletionResponseChunk] = stub.RequestChat(message)
        buffer = []
        for chunk in output_stream:
            buffer.append(chunk.content)
            if sum(map(len, buffer)) > 90:
                flush_buffer(buffer)
        if buffer:
            flush_buffer(buffer)

        while True:
            try:
                prompt = input('prompt: ')
                if prompt.startswith('.exit'):
                    break

                output_stream = stub.RequestChat(CompletionRequest(prompt=prompt))

                buffer = []
                for chunk in output_stream:
                    buffer.append(chunk.content)
                    if sum(map(len, buffer)) > 90:
                        flush_buffer(buffer)
                if buffer:
                    flush_buffer(buffer)

            except KeyboardInterrupt:
                break
