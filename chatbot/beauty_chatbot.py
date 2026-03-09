import chainlit as cl
import dotenv
import os
import re
 
from openai.types.responses import ResponseTextDeltaEvent
from agents import Runner, SQLiteSession
 
from beauty_agent import beauty_agent
from advanced_skin_analyzer import analyze_skin
 
dotenv.load_dotenv()
 
 
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    users_raw = os.getenv("CHAINLIT_USERNAME", "")
    users = {}
 
    for pair in users_raw.split(","):
        if ":" in pair:
            user, pwd = pair.split(":", 1)
            users[user.strip()] = pwd.strip()
 
    if username in users and users[username] == password:
        return cl.User(
            identifier=username,
            metadata={"role": "student", "provider": "credentials"},
        )
    return None
 
 
@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    user_id = user.identifier if user else "anonymous"
 
    # separate memory per authenticated user
    session = SQLiteSession(f"conversation_history_{user_id}")
    cl.user_session.set("agent_session", session)
 
    await cl.Message(
        content="""
👋 Hi! I'm your AI Beauty Assistant.
 
You can:
• Ask skincare questions
• Ask about ingredients
• Ask where to buy cosmetics
• Upload a selfie to analyze your skin
"""
    ).send()
 
 
@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")
 
    # IMAGE ANALYSIS
    if message.elements:
        image = message.elements[0]
        image_path = image.path
 
        await cl.Message(content="📷 Analyzing your skin...").send()
 
        analysis = analyze_skin(image_path)
 
        prompt = f"""
A computer vision system already analyzed the user's face.
 
DO NOT say you cannot analyze images.
 
Skin Analysis Report:
{analysis}
 
Tasks:
1. Identify the skin type
2. Explain detected skin conditions
3. Recommend a skincare routine
4. Suggest helpful ingredients
"""
 
        result = Runner.run_streamed(
            beauty_agent,
            prompt,
            session=session
        )
 
        msg = cl.Message(content="")
 
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                cleaned_delta = re.sub(
                    r"<thinking>.*?</thinking>",
                    "",
                    event.data.delta,
                    flags=re.DOTALL
                )
                if cleaned_delta:
                    await msg.stream_token(token=cleaned_delta)
 
            elif (
                event.type == "raw_response_event"
                and hasattr(event.data, "item")
                and hasattr(event.data.item, "type")
                and event.data.item.type == "function_call"
                and hasattr(event.data.item, "arguments")
                and event.data.item.arguments
            ):
                with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                    step.input = event.data.item.arguments
 
        await msg.update()
        return
 
    # NORMAL CHAT
    result = Runner.run_streamed(
        beauty_agent,
        message.content,
        session=session
    )
 
    msg = cl.Message(content="")
 
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)
 
        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and hasattr(event.data.item, "arguments")
            and event.data.item.arguments
        ):
            with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                step.input = event.data.item.arguments
 
    await msg.update()