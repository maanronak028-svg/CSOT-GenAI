import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
import re
from rank_bm25 import BM25Okapi
import aiohttp
import asyncio
import json
import random
import aiofiles
import webbrowser
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll, Container
from textual.widgets import Header, Footer, Input, RichLog, Static, Button, Label, LoadingIndicator, RadioButton, RadioSet
from textual.screen import ModalScreen
from textual.binding import Binding
from textual import work
from rich.markdown import Markdown
from typing import Any

load_dotenv()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "No Key")
gemini_client = genai.Client(api_key=GEMINI_KEY)

SERPER_KEY = os.environ.get("SERPER_API_KEY")

ALPHAXIV_MCP_URL = "https://api.alphaxiv.org/mcp/v1"
REDIRECT_URI = "http://localhost:8765/callback"
TOKEN_FILE = ".alphaxiv_tokens.json"
TUI_APP = None
aichat = None

nature_professional = "You are a highly capable, strictly professional AI assistant designed for productivity and technical support. Upon initialization, provide a formal introduction and state your readiness to assist the user with their tasks. Your primary objective is to deliver accurate, efficient, and objective information without conversational filler or emotional language. Maintain a formal, polite tone at all times. You must ensure your responses are well-structured."
nature_casual = "You are a super relaxed, casual, and friendly AI companion. Treat the user like a good friend, starting the conversation with a warm 'hey', a quick intro, and asking what they are up to or how you can help out. Keep the vibe light, use everyday conversational language, and feel free to show a bit of personality. You must keep your responses breezy and engaging. Avoid sounding robotic, corporate, or overly formal."
nature_blend = "You are a helpful AI assistant who blends a friendly, welcoming tone with professional reliability. Whenever a new conversation starts, warmly introduce yourself and politely ask the user how you can support them today. You must communicate clearly and efficiently. Keep the interaction focused, polite, and perfectly paced."

nature_expert = "You are an elite AI verification expert. Your sole purpose is to analyze, fact-check, and cross-verify answers provided by other AI assistants. When provided with a user's question and the previous AI's answer, evaluate the response for accuracy, logical consistency, and completeness. If the previous answer is correct, confirm it concisely. If the answer contains errors, point them out explicitly and provide the correct factual information. Maintain an objective, highly analytical tone."

search_guide = '''
VERY IMPORTANT - HALUCINATIONS PROTOCOL:

When using the web_search tool or web_fetch tool, you always have to return only the findings that you get from them. Don't and information on your own and only report what you find and what you are absolutely sure with.

Web Scouting & Triage Protocol:
When using the web_search tool, you will receive a list of URLs and snippets. You must evaluate the URLs for domain authority BEFORE calling the web_fetch tool.

Tier 1 (Mandatory Priority): Official documentation (e.g., docs.python.org, wireguard.com), primary source repositories (github.com), and educational/governmental domains (.edu, .gov).
Tier 2 (Acceptable): Known, highly-moderated technical hubs (e.g., StackOverflow) or reputable publisher blogs.
Tier 3 (IGNORE): Social media, user forums (Reddit, Quora), SEO content mills, and aggregators.

Constraint: Never fetch a Tier 3 URL unless Tier 1 and 2 sources completely fail to answer the query.
'''

class ChatBot:
    def __init__(self, model, message_limit, nature, app):
        self.app = app
        self.model = model
        self.message_limit = message_limit
        self.history = [{"role": "system", "content": search_guide + "\n" + nature}]
        self.conversations_count = 0
        self.search_tool = {
            "type": "function",
            "function": {
                "name": "tool_web_search",
                "description": (
                    "Scout the web for current information. Returns a lightweight list of search results with titles, URLs, and short snippets. "
                    "USE THIS FIRST to find relevant URLs. DO NOT rely on snippets for deep technical answers; use this tool "
                    "to find authoritative domains (.org, .edu, official docs), and then pass those URLs to 'tool_web_fetch'."
                    "CRITICAL EXCEPTION: Do NOT use this tool when you need to find papers/research papers, "
                    "you MUST use 'tool_discover_papers' instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Use targeted, highly specific keywords rather than conversational sentences (e.g., 'Navidrome port requirements' instead of 'What port does Navidrome use?').",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

        self.fetch_tool = {
            "type": "function",
            "function": {
                "name": "tool_web_fetch",
                "description": (
                    "Extracts and reads the deep content of web pages. Use this after 'tool_web_search' to read the actual content of the most authoritative URLs. "
                    "This tool bypasses javascript and strips out ads/navbars to return clean markdown data."
                    "CRITICAL EXCEPTION: Do NOT use this tool for arXiv.org links. If you need to read a research paper, "
                    "you MUST extract the ID and use 'tool_get_paper_content' instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "An array of full URLs to fetch (must include https://). Only pass authoritative or highly relevant URLs.",
                        },
                        "user_query": {
                            "type": "string",
                            "description": (
                                "OPTIONAL BUT HIGHLY RECOMMENDED. Provide 3-5 specific keywords here to trigger the internal BM25/Semantic extraction filter. "
                                "This forces the tool to return only the paragraphs relevant to your keywords, saving your context window. "
                                "Leave this blank ONLY if you need a general summary of the entire webpage."
                            ),
                        }
                    },
                    "required": ["urls"],
                },
            },
        }
        
        self.file_tool = {
            "type": "function",
            "function": {
                "name": "tool_local_file",
                "description": (
                    "Read from or write to a local file on the user's system. "
                    "Use 'read' to inspect code, config files, or logs. "
                    "Use 'write' to save scripts, update configurations, or output data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["read", "write"],
                            "description": "Whether to 'read' the file or 'write' to the file."
                        },
                        "filepath": {
                            "type": "string",
                            "description": "The absolute or relative path to the local file."
                        },
                        "content": {
                            "type": "string",
                            "description": "The exact content to write to the file. REQUIRED if operation is 'write'. Leave blank for 'read'."
                        }
                    },
                    "required": ["operation", "filepath"],
                },
            },
        }
        self.note_tool = {
            "type": "function",
            "function": {
                "name": "tool_save_research_note",
                "description": "Save research findings or notes to a markdown file in the 'notes/' folder for later sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the note, which will be used as the filename (e.g., 'navidrome_setup'). Do not include the .md extension."
                        },
                        "content": {
                            "type": "string",
                            "description": "The full markdown content of the research note to save."
                        }
                    },
                    "required": ["title", "content"]
                }
            }
        }
        self.tools_list = [self.search_tool, self.fetch_tool, self.file_tool, self.note_tool]
        self.gemini_tools = [
            {
                "function_declarations": [self.search_tool["function"], self.fetch_tool["function"], self.file_tool["function"], self.note_tool["function"]]
            }
        ]
        
    async def check_critic(self, prompt, cnt = 0):
        try:
            router_instruction = """You are an intent classification engine. Your only job is to analyze the user's text and determine if they are asking a new question or if they want to fact-check/verify the previous answer.

            Rules:
            - If the user is asking a new question, starting a new topic, or making a general statement, output strictly: 0
            - If the user is asking to verify, cross-check, confirm, or fact-check the previous answer (e.g., 'Are you sure?', 'Verify this', 'Is that true?'), output strictly: 1

            You must output ONLY the single digit 0 or 1. Absolutely no other text, spaces, or punctuation."""

            response = await gemini_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=router_instruction,
                    temperature=0.0
                )
            )
            
            result_text = response.text.strip()
            if int(result_text):
                await self.critic_verify(prompt)
            else:
                await self.get_response(prompt)
        except Exception as e:
            if cnt:
                await self.get_response(prompt)
            else:
                await self.check_critic(prompt, 1)
        
    async def get_response(self, prompt="", cnt=0):
        try:
            if prompt:
                self.history.append({"role": "user", "content": prompt})
                
            self.app.write_thought("[dim]Calling AI model...[/dim]")
            while True:
                stream = await client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    tools=self.tools_list,
                    tool_choice="auto",
                    stream=True
                )
                
                accumulated_text = ""
                tool_calls = []
                
                scroll = self.app.query_one("#chat-scroll", VerticalScroll)
                bubble = None
                content_widget = None
                prefix_text = "**Agent:** "
                
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            while len(tool_calls) <= tc.index:
                                tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            if tc.id:
                                tool_calls[tc.index]["id"] += tc.id
                            if tc.function and tc.function.name:
                                tool_calls[tc.index]["function"]["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
                                
                    if delta.content:
                        if bubble is None:
                            bubble = MessageBubble(Markdown(prefix_text), role="ai")
                            await scroll.mount(bubble)
                            content_widget = bubble.query(Static).first()
                            self.app._show_loading(False)
                        accumulated_text += delta.content
                        content_widget.update(Markdown(f"{prefix_text}{accumulated_text}"))
                        if "\n" in delta.content or " " in delta.content:
                            scroll.scroll_end(animate=False)
                
                if tool_calls:
                    message_tool_calls = []
                    for tc in tool_calls:
                        message_tool_calls.append({
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        })
                    self.history.append({"role": "assistant", "content": None, "tool_calls": message_tool_calls})
                    
                    for tool_call in message_tool_calls:
                        func_name = tool_call["function"]["name"]
                        args_str = tool_call["function"]["arguments"]
                        
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            args = {}
                            
                        self.app.write_thought(f"\n[bold cyan]Action:[/bold cyan] {func_name}")
                        if args:
                            display_args = args.copy()
                            if func_name in ("tool_local_file", "tool_save_research_note") and "content" in display_args:
                                display_args["content"] = f"<Skipped: {len(str(display_args['content']))} characters of content>"
                            self.app.write_thought(f"[bold cyan]Parameters:[/bold cyan] {json.dumps(display_args, indent=2)}")
                        
                        tool_output = ""
                        if func_name == "tool_web_search":
                            self.app.write_thought("[dim]Searching the web...[/dim]")
                            tool_output = await tool_web_search(args.get("query"))
                        elif func_name == "tool_web_fetch":
                            self.app.write_thought("[dim]Reading webpage...[/dim]")
                            tool_output = await tool_web_fetch(args.get("urls"), args.get("user_query", ""))
                        elif func_name == "tool_local_file":
                            self.app.write_thought(f"[dim]Performing {args.get('operation')} on {args.get('filepath')}...[/dim]")
                            tool_output = await tool_local_file(self.app, args.get("operation"), args.get("filepath"), args.get("content", ""))
                        elif func_name == "tool_save_research_note":
                            self.app.write_thought(f"[dim]Saving note '{args.get('title')}'...[/dim]")
                            tool_output = await tool_save_research_note(args.get("title"), args.get("content", ""))
                        elif func_name in self.mcp_tool_names:
                            self.app.write_thought(f"[dim]Executing MCP tool {func_name}...[/dim]")
                            tool_output = await execute_mcp_tool(func_name, args)
                            
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": func_name,
                            "content": str(tool_output)
                        })
                    continue 
                
                else:
                    if accumulated_text:
                        self.history.append({"role": "assistant", "content": accumulated_text})
                    scroll.scroll_end(animate=False)
                    break
                    
            self.conversations_count += 1
            if self.conversations_count >= self.message_limit:
                await self.summarize(False)
                
        except Exception as e:
            if cnt == 2:
                self.app.write_ai_message("Can't reach agent right now. Please try again.")
                self.app.write_thought(f"[red]Agent error after 3 attempts: {e}[/red]")
            else:
                await self.get_response("", cnt+1)
            
    async def summarize(self, print_output, silent=False, summarize_all=False):
        if len(self.history) > 2 or summarize_all:
            if not silent:
                self.app.write_thought("Summarizing messages till now. Please wait...")    
            try:
                if summarize_all:
                    temp = []
                    history_to_summarize = self.history
                else:
                    last_user_idx = 1
                    for i in range(len(self.history) - 1, 0, -1):
                        if self.history[i]["role"] == "user":
                            last_user_idx = i
                            break
                            
                    temp = self.history[last_user_idx:]
                    history_to_summarize = self.history[:last_user_idx]

                summary_instruction = "Generate a highly condensed, factual summary of the conversation up to this point. This summary will be used as a system memory state for an AI. Focus strictly on retaining established facts, user preferences, core context, and any ongoing tasks. Completely omit all conversational filler, greetings, and narrative flow. Output a dense, bulleted list of core data points."
                
                try:
                    transcript = ""
                    for msg in history_to_summarize[1:]:
                        role = msg.get('role', '').capitalize()
                        content = msg.get('content') or str(msg.get('tool_calls', 'Used a Tool'))
                        transcript += f"{role}: {content}\n"
                        
                    gemini_prompt = f"{summary_instruction}\n\nChat History:\n{transcript}"
                    
                    response = await gemini_client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=gemini_prompt
                    )
                    new_summary = response.text
                    
                except Exception as gemini_err:
                    if not silent:
                        self.app.write_thought(f"[System: Gemini summarize failed, falling back to OpenRouter. Error: {gemini_err}]")
                    
                    fallback_history = history_to_summarize.copy()
                    fallback_history.append({"role": "user", "content": summary_instruction})
                    
                    summary_response = await client.chat.completions.create(
                        model=self.model,
                        messages=fallback_history
                    )
                    new_summary = summary_response.choices[0].message.content
                self.history = [self.history[0]] + [{"role": "system", "content": new_summary}] + temp
                self.conversations_count = 1 if not summarize_all else 0                
                if not silent:
                    self.app.write_thought("Messages summarized.")
                if print_output:
                    self.app.write_ai_message("Summarized History: " + self.history[1]["content"])
                    
            except Exception as e:
                if not silent:
                    self.app.write_thought(f"Summarization completely failed!! Error: {e}")
        else:
            if print_output:
                if len(self.history) > 2:
                    self.app.write_ai_message("Summarized History: " + self.history[1]["content"])
                else:
                    self.app.write_thought("No History to Summarize.")
            
    async def critic_verify(self, question, cnt=0):
        if self.history[-1].get("role") == "assistant" and self.history[-1].get("content"):
            try:
                last_user_idx = 1
                for i in range(len(self.history) - 2, 0, -1): 
                    if self.history[i]["role"] == "user":
                        last_user_idx = i
                        break
                
                turn_messages = self.history[last_user_idx:]
                exchange_text = ""
                for msg in turn_messages:
                    role = msg.get("role", "").capitalize()
                    
                    if role == "User":
                        exchange_text += f"\n[User Question]\n{msg.get('content')}\n"
                        
                    elif role == "Assistant":
                        if msg.get("tool_calls"):
                            for tool in msg["tool_calls"]:
                                func_name = tool['function']['name']
                                args = tool['function']['arguments']
                                exchange_text += f"\n[AI Called Tool: {func_name}]\nArguments: {args}\n"
                        if msg.get("content"):
                            exchange_text += f"\n[Final AI Answer]\n{msg.get('content')}\n"
                            
                    elif role == "Tool":
                        exchange_text += f"\n[Tool Output ({msg.get('name')})]\n{msg.get('content')}\n"

                verification_prompt = f"""
                Please verify the following exchange. Evaluate if the Final AI Answer correctly and accurately addresses the User's question, given the tool outputs provided.
                If the AI hallucinated information not found in the tool output, point it out.
                
                --- START EXCHANGE ---
                {exchange_text}
                --- END EXCHANGE ---
                """
                
                self.app.write_thought("[dim]Verifier analyzing response...[/dim]")
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=verification_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=search_guide+"\n"+nature_expert,
                        temperature=0.2
                    )
                )
                
                output = response.text or ""
                if output:
                    await typewriter_to_ui(
                        app=self.app,
                        container_id="chat-scroll",
                        role_label="Verifier",
                        text=output,
                        role_class="verifier"
                    )
                
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": output})
                self.conversations_count += 1
                
                if self.conversations_count >= self.message_limit:
                    await self.summarize(False)
                    
            except Exception as e:
                if cnt==2:
                    self.app.write_ai_message("Can't reach Verifier right now. Please try again.")
                    self.app.write_thought(f"[red]Verifier error after 3 attempts: {e}[/red]")
                else:
                    await self.critic_verify(question, cnt+1)
        else:
            self.app.write_ai_message("Verifier: Can't find any final text response to verify.")
        
    async def do_exit(self):
        self.app.write_thought("Exiting...")
        await self.summarize(False, True, summarize_all=True)
        self.history = self.history[1:] + [{"role": "system", "content": "The user is ending the conversation. Generate a polite, warm, and professional closing message. Thank the user for their time, offer a brief well-wish for the rest of their day, and let them know you will be ready to help whenever they return. Keep the response concise, strictly between 2 to 3 sentences."}]
        await self.get_response()
        
class MessageBubble(Static):
    def __init__(self, content: Any, role: str = "ai", **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.message_content = content

    def compose(self) -> ComposeResult:
        yield Static(self.message_content, markup=True)

    def on_mount(self) -> None:
        self.add_class(f"message-{self.role}")

def sanitize_gemini_params(params):
    allowed = {"type", "properties", "required", "description", "enum", "items", "format"}
    if not isinstance(params, dict):
        return params
    cleaned = {}
    for k, v in params.items():
        if k not in allowed:
            continue
        if k == "properties" and isinstance(v, dict):
            cleaned[k] = {pk: sanitize_gemini_params(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            cleaned[k] = sanitize_gemini_params(v)
        else:
            cleaned[k] = v
    return cleaned

class SetupModal(ModalScreen[dict]):
    def compose(self) -> ComposeResult:
        yield Container(
            Label("[bold #8b8bff]Session Setup[/bold #8b8bff]", id="setup-title"),
            Label("[#c0c0e0]Which AI engine would you like to use?[/#c0c0e0]"),
            RadioSet(
                RadioButton("GPT OSS 120b", value=True),
                RadioButton("Owl Alpha"),
                RadioButton("GLM 4.5 Air"),
                id="model-select"
            ),
            Label("[#c0c0e0]Memory limit (messages before auto-summarize):[/#c0c0e0]"),
            Input(placeholder="5-15 (default: 10)", id="memory-input"),
            Label("[#c0c0e0]AI personality:[/#c0c0e0]"),
            RadioSet(
                RadioButton("Balanced Blend", value=True),
                RadioButton("Strictly Professional"),
                RadioButton("Friendly & Casual"),
                id="nature-select"
            ),
            Horizontal(
                Button("Launch", variant="success", id="btn-launch"),
                id="modal-buttons"
            ),
            id="setup-dialog"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-launch":
            model_map = {
                0: "openai/gpt-oss-120b:free",
                1: "openrouter/owl-alpha",
                2: "z-ai/glm-4.5-air:free",
            }
            nature_map = {
                0: nature_blend,
                1: nature_professional,
                2: nature_casual,
            }

            model_set = self.query_one("#model-select", RadioSet)
            model_idx = model_set.pressed_index if model_set.pressed_index >= 0 else 0

            nature_set = self.query_one("#nature-select", RadioSet)
            nature_idx = nature_set.pressed_index if nature_set.pressed_index >= 0 else 0

            mem_input = self.query_one("#memory-input", Input).value.strip()
            try:
                mem_limit = int(mem_input)
                mem_limit = max(5, min(15, mem_limit))
            except (ValueError, TypeError):
                mem_limit = 10

            self.dismiss({
                "model": model_map[model_idx],
                "nature": nature_map[nature_idx],
                "memory": mem_limit + 1,
            })

class PermissionModal(ModalScreen[bool]):
    def __init__(self, operation: str, filepath: str, content: str = ""):
        super().__init__()
        self.operation = operation
        self.filepath = filepath
        self.content = content

    def compose(self) -> ComposeResult:
        yield Container(
            Label(f"[bold red]SECURITY ALERT[/bold red]", id="alert-title"),
            Label(f"The AI wants to [bold]{self.operation.upper()}[/bold] the following file:"),
            Label(f"[cyan]{self.filepath}[/cyan]\n"),
            Label("Preview of changes:" if self.operation == "write" else "It will read the file contents."),
            RichLog(id="preview-log", auto_scroll=False),
            Horizontal(
                Button("Approve", variant="success", id="btn-approve"),
                Button("Deny", variant="error", id="btn-deny"),
                id="modal-buttons"
            ),
            id="dialog"
        )

    def on_mount(self) -> None:
        if self.operation == "write" and self.content:
            preview = self.query_one("#preview-log", RichLog)
            preview.write(self.content[:500] + ("\n...[truncated]" if len(self.content) > 500 else ""))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-approve":
            self.dismiss(True)
        else:
            self.dismiss(False)
            
class AgenticTUI(App):
    TITLE = "Dual-Agent AI"
    SUB_TITLE = "Agentic Research Assistant"

    CSS = """
    Screen {
        background: #0a0a0f;
    }

    Header {
        background: #12121a;
        color: #e0e0ff;
        dock: top;
        height: 3;
        content-align: center middle;
    }

    Footer {
        background: #12121a;
        color: #6e6e8a;
    }

    #main-layout {
        height: 1fr;
        margin: 0;
        padding: 0;
    }

    #chat-panel {
        width: 1fr;
        height: 1fr;
        margin: 0 0 0 1;
        background: #0f0f18;
        border: solid #1e1e3a;
        border-title-color: #8b8bff;
        border-title-style: bold;
    }

    #chat-scroll {
        height: 1fr;
        padding: 1 2;
        scrollbar-color: #3a3a6a;
        scrollbar-color-hover: #5a5aaa;
        scrollbar-color-active: #7a7aff;
        scrollbar-background: #0f0f18;
    }

    #sidebar-panel {
        width: 32%;
        height: 1fr;
        margin: 0 1 0 0;
        background: #0f0f18;
        border: solid #1e3a1e;
        border-title-color: #6aff6a;
        border-title-style: bold;
    }

    #thought-log {
        height: 1fr;
        padding: 1;
        scrollbar-color: #2a4a2a;
        scrollbar-color-hover: #3a6a3a;
        scrollbar-background: #0f0f18;
    }

    .message-ai {
        background: #161630;
        color: #d0d0f0;
        border: solid #2a2a5a;
        padding: 1 2;
        margin: 1 8 1 0;
        width: auto;
    }

    .message-user {
        background: #1a2a1a;
        color: #d0f0d0;
        border: solid #2a4a2a;
        padding: 1 2;
        margin: 1 0 1 8;
        text-align: right;
    }

    .message-verifier {
        background: #2a1a2a;
        color: #f0d0f0;
        border: solid #4a2a4a;
        padding: 1 2;
        margin: 1 4 1 4;
    }

    .message-system {
        background: #1a1a10;
        color: #c0c080;
        border: solid #3a3a20;
        padding: 1 2;
        margin: 1 12 1 12;
        text-align: center;
        text-style: italic;
    }

    .role-label-ai {
        color: #8b8bff;
        text-style: bold;
        margin-bottom: 0;
    }

    .role-label-user {
        color: #6aff6a;
        text-style: bold;
        margin-bottom: 0;
    }

    .role-label-verifier {
        color: #ff6aff;
        text-style: bold;
        margin-bottom: 0;
    }

    #input-area {
        dock: bottom;
        height: auto;
        padding: 0 1 1 1;
        background: #0a0a0f;
    }

    #input-box {
        border: tall #2a2a5a;
        background: #12121a;
        color: #e0e0ff;
        padding: 0 1;
    }

    #input-box:focus {
        border: tall #5a5aff;
    }

    #loading-indicator {
        height: 1;
        margin: 0 1;
        color: #5a5aff;
        display: none;
    }

    #loading-indicator.visible {
        display: block;
    }

    #dialog {
        padding: 2 4;
        width: 60%;
        height: 60%;
        border: thick #2a2a5a;
        background: #12121a;
        align: center middle;
    }

    #setup-dialog {
        padding: 2 4;
        width: 55%;
        height: auto;
        max-height: 80%;
        border: thick #2a2a5a;
        background: #12121a;
        align: center middle;
    }

    #setup-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #setup-dialog RadioSet {
        margin: 0 0 1 2;
        background: #0f0f18;
        height: auto;
    }

    #setup-dialog Input {
        margin: 0 0 1 2;
        width: 30;
    }

    #alert-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #preview-log {
        height: 1fr;
        border: solid #2a2a5a;
        margin: 1 0;
        background: #0a0a0f;
    }

    #modal-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 2;
        min-width: 16;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+s", "summarize", "Force Summarize", priority=True),
        Binding("ctrl+l", "clear_thoughts", "Clear Brain Panel", priority=True),
        Binding("ctrl+k", "clear_chat", "Clear Chat", priority=True)
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-panel"):
                yield VerticalScroll(id="chat-scroll")

            with Vertical(id="sidebar-panel"):
                yield RichLog(id="thought-log", markup=True, wrap=True)

        with Vertical(id="input-area"):
            yield LoadingIndicator(id="loading-indicator")
            yield Input(placeholder="Type your message here... (Enter to send)", id="input-box")
        yield Footer()

    async def on_mount(self) -> None:
        global TUI_APP
        TUI_APP = self
        self.query_one("#chat-panel").border_title = "Chat"
        self.query_one("#sidebar-panel").border_title = "AI Brain"
        self.mcp_schemas = await fetch_mcp_tools()
        self.push_screen(SetupModal(), callback=self._on_setup_done)

    def _on_setup_done(self, result: dict) -> None:
        self.query_one(Input).focus()
        self.start_session(result)

    @work(exclusive=True)
    async def start_session(self, config: dict) -> None:
        global aichat
        self.write_system_message("Setting up session...")
        model = config["model"]
        nature = config["nature"]
        memory = config["memory"]
        aichat = ChatBot(model=model, message_limit=memory, nature=nature, app=self)
        if self.mcp_schemas:
            aichat.tools_list.extend(self.mcp_schemas)
            for schema in self.mcp_schemas:
                cleaned_params = sanitize_gemini_params(schema["function"].get("parameters", {}))
                gemini_decl = {
                    "name": schema["function"]["name"],
                    "description": schema["function"].get("description", ""),
                    "parameters": cleaned_params,
                }
                aichat.gemini_tools[0]["function_declarations"].append(gemini_decl)
            aichat.mcp_tool_names = [schema["function"]["name"] for schema in self.mcp_schemas]
            self.write_thought(f"[green]Loaded {len(aichat.mcp_tool_names)} tools from AlphaXiv[/green]")
        else:
            aichat.mcp_tool_names = []
        self.write_system_message(f"Ready! Model: {model.split('/')[-1]} | Memory: {memory - 1} | Say hello to begin.")
        self._show_loading(True)
        await aichat.get_response("")
        
    def write_ai_message(self, text: str) -> None:
        self._show_loading(False)
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        bubble = MessageBubble(Markdown(f"**Agent:** {text}"), role="ai")
        scroll.mount(bubble)
        scroll.scroll_end(animate=False)

    def write_user_message(self, text: str) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        bubble = MessageBubble(Markdown(f"**You:** {text}"), role="user")
        scroll.mount(bubble)
        scroll.scroll_end(animate=False)

    def write_system_message(self, text: str) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        bubble = MessageBubble(f"[italic #c0c080]{text}[/italic #c0c080]", role="system")
        scroll.mount(bubble)
        scroll.scroll_end(animate=False)

    def write_thought(self, text: str) -> None:
        self.query_one("#thought-log", RichLog).write(text)

    def _show_loading(self, show: bool) -> None:
        loader = self.query_one("#loading-indicator", LoadingIndicator)
        if show:
            loader.add_class("visible")
        else:
            loader.remove_class("visible")

    def action_clear_thoughts(self) -> None:
        self.query_one("#thought-log", RichLog).clear()

    def action_clear_chat(self) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.remove_children()
        self.write_system_message("Chat cleared.")

    @work(exclusive=True)
    async def action_quit(self) -> None:
        if aichat and aichat.conversations_count > 0:
            self.write_system_message("Generating farewell...")
            self._show_loading(True)
            await aichat.do_exit()
            self._show_loading(False)
            await asyncio.sleep(2)
        self.app.exit()
        
    @work(exclusive=True)
    async def action_summarize(self) -> None:
        self.write_system_message("Forcing summarization...")
        await aichat.summarize(True)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value
        if not user_text.strip():
            return
        event.input.value = ""
        self.write_user_message(user_text)
        self.query_one("#thought-log", RichLog).clear()
        self.write_thought(f"[dim]Analyzing user query...[/dim]")
        self._show_loading(True)
        self.process_ai_response(user_text)

    @work(exclusive=True)
    async def process_ai_response(self, user_text: str) -> None:
        await aichat.check_critic(user_text)
        self._show_loading(False)
        
async def typewriter_to_ui(app, container_id, role_label, text, role_class="ai", speed=0.0048):
    prefix_text = f"**{role_label}:** "

    scroll = app.query_one(f"#{container_id}", VerticalScroll)
    bubble = MessageBubble(Markdown(prefix_text), role=role_class)
    await scroll.mount(bubble)
    content_widget = bubble.query(Static).first()

    app._show_loading(False)

    displayed = ""
    chunk_size = max(1, int(0.015 / speed))
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        displayed += chunk
        content_widget.update(Markdown(f"{prefix_text}{displayed}"))
        if "\n" in chunk or " " in chunk:
            scroll.scroll_end(animate=False)
        await asyncio.sleep(speed * len(chunk))
    scroll.scroll_end(animate=False)
        
class FileTokenStorage(TokenStorage):
    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None
        if os.path.exists(TOKEN_FILE):
            try:
                data = json.loads(open(TOKEN_FILE).read())
                if data.get("tokens"):
                    self.tokens = OAuthToken(**data["tokens"])
                if data.get("client_info"):
                    self.client_info = OAuthClientInformationFull(**data["client_info"])
            except Exception:
                pass

    def _save(self):
        data = {}
        if self.tokens:
            data["tokens"] = self.tokens.model_dump(mode="json")
        if self.client_info:
            data["client_info"] = self.client_info.model_dump(mode="json")
        open(TOKEN_FILE, "w").write(json.dumps(data, indent=2))

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens
        self._save()

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info
        self._save()
        


async def open_browser(auth_url: str) -> None:
    if TUI_APP:
        TUI_APP.write_system_message(f"Opening browser for AlphaXiv login...")
        TUI_APP.write_thought(f"[yellow]Auth URL: {auth_url}[/yellow]")
    webbrowser.open(auth_url)

async def wait_for_callback() -> tuple[str, str | None]:
    code = state = None
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal code, state
            params = parse_qs(urlparse(self.path).query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authorized. You can close this tab and return to the terminal.</h1>")
        def log_message(self, *args):
            pass 

    server = HTTPServer(("localhost", 8765), Handler)
    server.timeout = 120
    
    await asyncio.to_thread(server.handle_request)
    server.server_close()

    if not code:
        raise RuntimeError("OAuth callback received no authorization code.")
    return code, state

auth_provider = OAuthClientProvider(
    server_url=ALPHAXIV_MCP_URL,
    client_metadata=OAuthClientMetadata(
        client_name="ResearchBot AlphaXiv Integration",
        redirect_uris=[REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="read",
    ),
    storage=FileTokenStorage(),
    redirect_handler=open_browser,
    callback_handler=wait_for_callback,
)

async def fetch_single_url(session, url, user_query=""):
    def process_markdown(markdown_text):
        if len(markdown_text) > 2500:
            if user_query:
                return bm25_filter(markdown_text, user_query)
            else:
                return gemini_filter(markdown_text)
        return markdown_text

    base_url = url.rstrip('/')
    llms_url = f"{base_url}/llms.txt"
    
    try:
        short_timeout = aiohttp.ClientTimeout(total=3)
        async with session.get(llms_url, timeout=short_timeout) as txt_response:
            if txt_response.status == 200:
                markdown = await txt_response.text()
                if TUI_APP:
                    TUI_APP.write_thought(f"[System: llms.txt shortcut found at {llms_url}]")
                return {"url": url, "content": process_markdown(markdown)}
    except Exception:
        pass
        
    jina_url = "https://r.jina.ai/" + url
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(jina_url, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            markdown = await response.text()
            return {"url": url, "content": process_markdown(markdown)}        
    except Exception as e:
        return {"url": url, "content": f"Failed to fetch: {str(e)}"}
    
fetch_semaphore = asyncio.Semaphore(3)
async def fetch_single_url_politely(session, url, user_query=""):
    async with fetch_semaphore:
        await asyncio.sleep(random.uniform(0.5, 1.5))
        return await fetch_single_url(session, url, user_query)

async def tool_web_fetch(urls, user_query=""):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_url_politely(session, url, user_query) for url in urls]
        results = await asyncio.gather(*tasks)
        return str(results)

def bm25_filter(markdown_text, user_query, char_limit=2000):
    raw_chunks = [chunk.strip() for chunk in markdown_text.split('\n\n') if len(chunk.strip()) > 40]
    
    if not raw_chunks:
        return "Couldn't fetch URL data, the URL is probably broken, please try other URLs."

    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    tokenized_chunks = [tokenize(chunk) for chunk in raw_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = tokenize(user_query)    
    scores = bm25.get_scores(tokenized_query)
    best_score = max(scores)
    
    if best_score > 2.0:
        scored_chunks = sorted(zip(scores, raw_chunks), key=lambda x: x[0], reverse=True)
        condensed_context = ""
        for score, chunk in scored_chunks:
            if score <= 0:
                break    
            addition = f"\n\n...[snip]...\n\n{chunk}" if condensed_context else chunk
            if len(condensed_context) + len(addition) > char_limit and condensed_context != "":
                break     
            condensed_context += addition    
        return condensed_context
    else:
        return gemini_filter(markdown_text, user_query)
    
def gemini_filter(markdown_text, user_query="", cnt=0):
    try:
        safe_text = markdown_text[:40000]
        if not user_query:
            prompt = f"""You are an elite data extraction agent. Analyze the following webpage markdown and generate a highly dense, factual summary. 
Focus strictly on core concepts, technical specifications, or primary arguments. Completely omit navigational text, ads, and conversational filler. 
Format your response as a bulleted list. 
CRITICAL CONSTRAINT: Your entire response MUST strictly be under 2000 characters.

Webpage Markdown:
{safe_text}"""
        else:
            prompt = f"""You are an elite research agent. Scan the following webpage markdown and extract ONLY the information explicitly relevant to the user's query.
Ignore all other context. If the answer is not present, state clearly that the webpage does not contain the requested information.
Format the extracted data clearly and concisely.
CRITICAL CONSTRAINT: Your entire response MUST strictly be under 2000 characters.

User Query: {user_query}

Webpage Markdown:
{safe_text}"""
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        if not cnt:
            time.sleep(2)
            return gemini_filter(markdown_text, user_query, cnt+1)
        else:
            return f"[SYSTEM WARNING: Semantic filtering failed.]\n\n{markdown_text[:2500]}"
        
async def tool_web_search(query, num_results=5):
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query,
        "num": num_results
    })
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload, timeout=5) as response:
                response.raise_for_status()
                data = await response.json()
                organic_results = data.get("organic", [])
                formatted_results = []
                for idx, result in enumerate(organic_results):
                    formatted_results.append(
                        f"{idx+1}. Title: {result.get('title')}\n"
                        f"URL: {result.get('link')}\n"
                        f"Snippet: {result.get('snippet')}\n"
                    )
                return "\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {str(e)}"
    
async def tool_local_file(app, operation, filepath, content=""):
    try:
        approved = await app.push_screen_wait(PermissionModal(operation, filepath, content))
        if not approved:
            return f"[System: Access Denied. The user refused permission.]"
        if operation == "read":
            if not os.path.exists(filepath):
                return f"Error: File '{filepath}' does not exist."
                
            async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
                data = await f.read()
                if len(data) > 10000:
                    return f"[Warning: File too large. Returning first 10000 characters.]\n\n{data[:10000]}"
                return data

        elif operation == "write":
            true_path = os.path.abspath(filepath)
            os.makedirs(os.path.dirname(true_path), exist_ok=True)
            
            async with aiofiles.open(true_path, mode='w', encoding='utf-8') as f:
                await f.write(content)
            return f"Success: Successfully wrote {len(content)} characters to {true_path}."
            
        else:
            return "Error: Invalid operation. Must be 'read' or 'write'."
            
    except Exception as e:
        return f"File operation failed: {str(e)}"

async def tool_save_research_note(title: str, content: str) -> str:
    try:
        os.makedirs("notes", exist_ok=True)
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        if not safe_title:
            safe_title = "untitled_note"
        filepath = os.path.join("notes", f"{safe_title}.md")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Note successfully saved to {filepath}."
    except Exception as e:
        return f"Error saving note: {str(e)}"

    
async def fetch_mcp_tools():
    try:
        async with httpx.AsyncClient(auth=auth_provider, follow_redirects=True, timeout=60) as http:
            async with streamable_http_client(ALPHAXIV_MCP_URL, http_client=http) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_tools = await session.list_tools()
                    
                    openai_tools = []
                    for tool in mcp_tools.tools:
                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema, 
                            },
                        })
                    return openai_tools
    except Exception as e:
        err = e.exceptions[0] if hasattr(e, "exceptions") else e
        if TUI_APP:
            TUI_APP.write_thought(f"\n[System Warning: Could not connect to AlphaXiv MCP: {err}]")
        return []

async def execute_mcp_tool(tool_name, args):
    try:
        async with httpx.AsyncClient(auth=auth_provider, follow_redirects=True, timeout=60) as http:
            async with streamable_http_client(ALPHAXIV_MCP_URL, http_client=http) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=args)
                    
                    if result.content and result.content[0].type == "text":
                        return result.content[0].text
                    return "Tool executed successfully, but returned no text."
    except Exception as e:
        err = e.exceptions[0] if hasattr(e, "exceptions") else e
        return f"MCP Tool execution failed: {str(err)}"

if __name__ == "__main__":
    app = AgenticTUI()
    app.run()