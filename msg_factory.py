from typing import Optional, Union, List, Literal

from llama_cpp import ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPart, \
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestAssistantMessage, \
    ChatCompletionRequestToolMessage, ChatCompletionRequestFunctionMessage, ChatCompletionRequestMessageContentPartText, \
    ChatCompletionRequestMessageContentPartImageImageUrl, ChatCompletionRequestMessageContentPartImage

MessageContent = Optional[Union[
    str,
    List[ChatCompletionRequestMessageContentPart]
]]

"""
@staticmethod vs __new__ vs plain old functions?
"""
def create_message(
        role: Union[
            Literal['system'],
            Literal['user'],
            Literal['assistant'],
            Literal['tool'],
            Literal['function'],
        ],
        content: MessageContent,
        **kwargs,
) -> ChatCompletionRequestMessage:
    match role:
        case 'system':
            return ChatCompletionRequestSystemMessage(
                role='system',
                content=content,
            )
        case 'user':
            return ChatCompletionRequestUserMessage(
                role='user',
                content=content,
            )
        case 'assistant':
            return ChatCompletionRequestAssistantMessage(
                role='assistant',
                content=content,
                **kwargs,
            )
        case 'tool':
            if 'tool_call_id' not in kwargs:
                raise ValueError('tool_call_id')
            return ChatCompletionRequestToolMessage(
                role='tool',
                content=content,
                **kwargs,
            )
        case 'function':
            if 'name' not in kwargs:
                raise ValueError('name')
            return ChatCompletionRequestFunctionMessage(
                role='function',
                content=content,
                name=kwargs.get('name'),
            )
    raise ValueError(f'invalid role: {role}')


def create_content(
    content_type: Union[
        Literal['text'],
        Literal['image_url'],
    ],
    text: Optional[str] = None,
    img: Optional[Union[
        str,
        ChatCompletionRequestMessageContentPartImageImageUrl
    ]] = None,
) -> ChatCompletionRequestMessageContentPart:
    match content_type:
        case 'text':
            if not text:
                raise ValueError('content is missing')
            return ChatCompletionRequestMessageContentPartText(
                type='text',
                text=text,
            )
        case 'image_url':
            if not img:
                raise ValueError('content is missing')
            return ChatCompletionRequestMessageContentPartImage(
                type='image_url',
                image_url=img,
            )
    raise ValueError(f'invalid type: {content_type}')


ChatMessage = create_message
ChatMessageContent = create_content
