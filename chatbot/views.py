from django.shortcuts import render
from typing import Iterator
from django.shortcuts import render, redirect
from django.http import JsonResponse,  HttpResponse, StreamingHttpResponse
import requests
import json
import time
from .models import Chat
from django.template.response import TemplateResponse
from django.db import transaction
from django.utils import timezone

FLASK_APP_ENDPOINT = "https://zep.hcmute.fit/7510/generate"

# def generate(message):
#     messages = [{"role": "user", "content": message}]
    
#     data = {
#         "prompt": messages,
#         "stream": True 
#     }

#     try:
#         # Request with streaming
#         response = requests.post(FLASK_APP_ENDPOINT, json=data, stream=True, timeout=300)

#         if response.status_code != 200:
#             return f"Error: Received status code {response.status_code} from the server"

#         full_response = ""
        
#         for chunk in response.iter_lines():
#             if chunk:
#                 decoded_chunk = chunk.decode('utf-8')
#                 json_data = json.loads(decoded_chunk[6:])
#                 full_response += json_data.get('response', '')
#                 print(json_data.get('response', '')) 

#         return full_response

#     except requests.RequestException as e:
#         return f"Error: {e}"


def chatbot(request):
    if not request.user.is_authenticated:
        return HttpResponse('Unauthorized', status=401)

    chats = Chat.objects.filter(user=request.user).order_by('created_at')

    if request.method == 'POST':
        message = request.POST.get('message')
        is_mini_chat = request.POST.get('is_mini_chat', 'false') == 'true'

        def stream_response():
            messages = [{"role": "user", "content": message}]
            data = {
                "prompt": messages,
                "stream": True
            }

            try:
                response = requests.post(
                    FLASK_APP_ENDPOINT, json=data, stream=True, timeout=300
                )
                if response.status_code != 200:
                    yield f"Error: Server returned status code {response.status_code}\n\n"
                    return

                full_response = ""

                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8')
                        if decoded_chunk.startswith("data:"):
                            decoded_chunk = decoded_chunk[5:]
                        try:
                            json_data = json.loads(decoded_chunk)
                            response_part = json_data.get("response", "")
                            full_response += response_part
                            yield response_part
                        except json.JSONDecodeError:
                            yield "Error decoding JSON\n\n"
                            return

                # Lưu chat sau khi nhận toàn bộ response
                with transaction.atomic():
                    Chat.objects.create(
                        user=request.user,
                        message=message,
                        response=full_response.strip(),
                        created_at=timezone.now()
                    )

            except requests.RequestException as e:
                yield f"Error: {e}\n\n"

        if is_mini_chat:
            return StreamingHttpResponse(stream_response(), content_type="text/event-stream")
        else:
            return StreamingHttpResponse(stream_response(), content_type="text/event-stream")

    return render(request, 'chatbot/chatbot.html', {'chats': chats})

def chat_with_bot(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    if request.method == 'POST':
        message = request.POST.get('message')
        
        def stream_response():
            messages = [{"role": "user", "content": message}]
            data = {
                "prompt": messages,
                "stream": True
            }

            try:
                response = requests.post(
                    FLASK_APP_ENDPOINT, json=data, stream=True, timeout=300
                )
                if response.status_code != 200:
                    yield f"Error: Server returned status code {response.status_code}\n\n"
                    return

                full_response = ""

                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8')
                        if decoded_chunk.startswith("data:"):
                            decoded_chunk = decoded_chunk[5:]
                        try:
                            json_data = json.loads(decoded_chunk)
                            response_part = json_data.get("response", "")
                            full_response += response_part
                            yield response_part
                        except json.JSONDecodeError:
                            yield "Error decoding JSON\n\n"
                            return

                # Lưu chat sau khi nhận toàn bộ response
                with transaction.atomic():
                    Chat.objects.create(
                        user=request.user,
                        message=message,
                        response=full_response.strip(),
                        created_at=timezone.now()
                    )

            except requests.RequestException as e:
                yield f"Error: {e}\n\n"

        return StreamingHttpResponse(stream_response(), content_type="text/event-stream")

    return JsonResponse({'error': 'Method not allowed'}, status=405)