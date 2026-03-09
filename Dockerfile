FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["chainlit", "run", "chatbot/beauty_chatbot.py", "--host", "0.0.0.0", "--port", "7860"]
