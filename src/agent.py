"""
The agent: a Groq-hosted LLM (Llama 3.3 70B) with tool-calling that decides
which of our three tools to call to answer a question about Hormuz vessel
traffic, executes it, and synthesizes a natural-language answer.

Requires GROQ_API_KEY in a local .env file.
Usage:
    python src/agent.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from groq import Groq, BadRequestError

from agent_tools import get_flagged_vessels, get_vessel_history, explain_flag, find_vessel_by_name

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a maritime intelligence analyst assistant for the Strait of Hormuz. "
    "You have tools to look up flagged vessels, a vessel's position history, and "
    "why a specific vessel was flagged. Use the tools to ground every factual claim -- "
    "never invent vessel names, MMSIs, or reasons. If a question needs more than one "
    "tool call to answer fully (e.g. 'which flagged vessel has the longest gap and where "
    "was it before'), call the tools in sequence before giving your final answer. "
    "Never call a tool with a guessed or placeholder argument (like 'unknown' or a made-up "
    "MMSI) -- if you don't yet know a required argument such as an MMSI, call the tool that "
    "discovers it first. If you know a vessel's name but not its MMSI, always call "
    "find_vessel_by_name first to resolve it -- never guess or construct an MMSI from a name. "
    "Call at most one tool per turn and wait for its actual result before calling a second, "
    "dependent tool -- never write one function call as the argument value of another. "
    "If a tool returns an error saying an MMSI does not exist, report that directly to the "
    "user -- do not then guess a vessel name to search for unless the user actually gave one. "
    "Keep answers concise and factual, like a briefing, not a chatbot."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_flagged_vessels",
            "description": "Get all vessels currently flagged by the anomaly detectors (AIS gaps, loitering, rendezvous).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vessel_history",
            "description": "Get the full position history for a specific vessel by MMSI.",
            "parameters": {
                "type": "object",
                "properties": {"mmsi": {"type": "string", "description": "The vessel's MMSI number"}},
                "required": ["mmsi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_flag",
            "description": "Get the specific reason(s) a vessel was flagged, by MMSI.",
            "parameters": {
                "type": "object",
                "properties": {"mmsi": {"type": "string", "description": "The vessel's MMSI number"}},
                "required": ["mmsi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_vessel_by_name",
            "description": "Look up a vessel's MMSI and flag state by its name. Use this whenever you know a vessel's name but not its MMSI -- never guess an MMSI from a name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "The vessel's name, or part of it"}},
                "required": ["name"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_flagged_vessels": get_flagged_vessels,
    "get_vessel_history": get_vessel_history,
    "explain_flag": explain_flag,
    "find_vessel_by_name": find_vessel_by_name,
}


def ask(question: str, max_tool_hops: int = 4) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_tool_hops):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
            )
        except BadRequestError as e:
            if "tool_use_failed" in str(e):
                print("  [note] model produced a malformed tool call -- retrying once")
                try:
                    response = client.chat.completions.create(
                        model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
                    )
                except BadRequestError:
                    return ("I had trouble forming a valid tool call for this question -- "
                            "could you try rephrasing it, e.g. asking about one vessel at a time?")
            else:
                raise
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for call in msg.tool_calls:
            fn = AVAILABLE_FUNCTIONS[call.function.name]
            args = json.loads(call.function.arguments or "{}")
            if args is None:
                args = {}
            print(f"  [tool call] {call.function.name}({args})")
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })

    return "I couldn't resolve this within the allowed number of tool calls."


if __name__ == "__main__":
    print("Dark Fleet Watch analyst -- ask a question (or 'quit')\n")
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        answer = ask(q)
        print(f"\n{answer}\n")
