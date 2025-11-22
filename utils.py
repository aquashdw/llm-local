from typing import List

from transformers import LlamaTokenizer

from msg_factory import ChatMessage


def flush_buffer(buffer):
    joined = ''.join(buffer)
    print(''.join(buffer))
    buffer.clear()
    return joined


def forget_messages(tokenizer: LlamaTokenizer, token_limit: int, messages: List[ChatMessage]):
    token_counts = []
    for message in messages:
        if 'content' not in message:
            continue
        content = message.get('content')
        if isinstance(content, str):
            token_counts.append(len(tokenizer.tokenize(message.get('content').encode('utf-8'))))
        elif isinstance(content, list):
            for content_part in content:
                if content_part.get('type') != 'text':
                    continue
                token_counts.append(len(tokenizer.tokenize(content_part.get('text').encode('utf-8'))))
    token_total = sum(token_counts)
    token_ceil = token_limit
    if token_total <= token_ceil:
        # print('forget--------')
        # print(f'token count within range: {token_total}')
        # print('forget--------')
        return messages

    new_messages = [messages[0]]
    token_removed = 0
    idx = 1
    while idx < len(token_counts) and token_total - token_removed > token_ceil:
        token_removed += token_counts[idx]
        token_removed += token_counts[idx + 1]
        idx += 2

    new_messages.extend(messages[idx:])
    # print('forget--------')
    # print(f'removed total {idx - 1} messages to remove {token_removed} tokens')
    # print(f'now: {token_total}')
    # print('forget--------')
    return new_messages
