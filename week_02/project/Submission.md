### Intialization and Features of the Agent

## Startup

1) Create a .env file with all the api keys given in .env.example(GEMINI_API_KEY is optional but recommended.)
2) Open the directory/folder which has agent.py and requirements.txt and run this command to install all the requirements-

```bash
pip install -r requirements.txt
```

3) Once all dependencies are installed, run this command to start the Agent-

```bash
python agent.py
```

## Features

- **Split-Panel TUI and great UI**: It has good-looking and user friendly, dark-neon themed UI, with a `Split-Panel TUI`, one panel for User and AI/Verifier chat and other panel to show inner thoughts of the AI, different coloured bubbles covering user prompts and AI/Verifier responses and system messages along with left and right alignment for better seperation. The chat panel has a very illustrative `markdown support` so that AI generated responses can be understood easily and look good.

- **Triple-Model Selection, Setting Memory Limit and Nature**: On start, you can choose which model you want to use for the agent from the three available options, set the memory limit before auto summarization of messages in the range 5-15 and also select the nature of agent from three different options.

- **Prompt Checking**: Using Gemini to smartly check if user wants to get an answer from agent or if he wants to verify previous assistant's response. Implemented in `check_critic` method of `ChatBot` class. Though it requires `GEMINI_API_KEY` otherwise it will fail silently and directly give response from Agent.

- **Clear Instructions and Reduced Hallucinations**: All the tasks done by the AI have big and clear prompts for the Agent/Verifier so that it does the intended task perfectly. Hallucinations from `web_search` and `web_fetch` or any other reasons are controlled and reduce by strict instructions. Further hallucinations can be checked and cross-verified from `Verifier`.

- **Verifier**: Automatically responds if the user is asking to verify the `Agent`'s last answer, checks the agents thinking and tools used and cross-verifies the facts to verify `Agent`'s last answer. Requires `GEMINI_API_KEY`.

- **Summary Generation**: Auto summarizes messages after the conversation reaches memory limit to save API tokens and credits and get faster responses. Checks if all messages are already summarized to save time and tokens. Auto summarize saves last response from AI and the tools used for Verifier to work easily. Uses `gemini-2.5-flash` for fast response but fallbacks to user chosen OpenRouter model in case of missing API key or limit reached or high-demand or any other error.

- **Async Handling**: Responses by AI are handled `asynchronously` and can be stopped or interrupted in between if user gives another prompt. Generation of `summary` is also handled in background.

- **Error Handling**: The `Agent` has complete error handling and uses try-except blocks to prevent crashes. It is completly crash free except for api key errors in .env file(`GEMINI_API_KEY` is optional though) and takes in account all edge cases and system/tools possible failures. Also has fallbacks and multiple tries with delays to account for high-demand errors and model unavailable errors.

- **Clean Exit**: During exit, all messages are `summarized` and an exit message is generated based on the chat summary to get a graceful exit.

## Custom Shortcuts for TUI

- **Ctrl + q**: `^q` to `exit`.

- **Ctrl + k**: `^k` to clear `Chat Panel`.

- **Ctrl + l**: `^l` to clear `AI Brain Panel`.

- **Ctrl + s**: `^s` to force generate `summary`.

## Tools

- **web-search**: Uses serper to search the web and get information needed by AI.

- **web-fetch**: Checks for llms.txt first, if found uses it to get markdown, else, uses a very good and useful site `r.jina.ai` to efficiently and most-productively generate markdown of HTML pages, even those pages that have `Dynamic pages` and run on `javascript`. Then this markdown text is passed through BM25 filter if it is very large to shorten it and get only required part. If BM25 filter doesn't result in a good filtered data or if we want full summary, then we use gemini model(requires `GEMINI_API_KEY`) to generate summary and if even that fails due to some errors then it gets starting 2500 characters of that markdown to get introduction and its general summary. Also supports multiple URL fetch at the same time `asynchronously` and has safety measure to prevent too many requests error.

- **AlphaXiv Tools**: Uses OAuth to load AlphaXiv tools like `discover_papers` and `get_paper_content` to easily search and fetch information from research papers.

- **Files Read-Write with Safety Permissions**: Can read and write files relative to current directory of terminal or absolute address, also, before reading and writing to a file, it always asks for user permission to read or make changes to that file using a popup to add an extra safety layer and prevent the AI from accessing sensitive files.

- **Save Research Papers**: Saves any research that it did during the session and all the information it found in a notes/ directory relative to current directory location of the terminal on user's command.

## Difficulties Faced

- **Web-Fetching**: Countering the token cost and memory limit issues due to very large markdown files was a very difficult task, also, getting markdown of javascript only pages and Dynamic Pages without using `playwright`(as it is very heavy) also required a lot of research before I found jina ai.

- **Error-Handling and Edge-Cases**: It was a very hard task to handle all the errors and edge cases and find their solutions and also required many many test runs and many gemini api token exhaustions.