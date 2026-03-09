#!/usr/bin/env bash
chainlit run chatbot/beauty_chatbot.py --host 0.0.0.0 --port ${PORT:-8000}
