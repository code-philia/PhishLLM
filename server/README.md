# PhishLLM Interface

Interface architecture
- client: HTML, CSS, JS with picoCSS for styling
- server: Flask as a bridge between PhishLLM and client

Overview
- `/static`: JS files for client interaction
- `/templates`: static client UI
- `announcer.py`: handles responses, act as messaging bridge between PhishLLM and client
- `server.py`: handles requests from client

## Announcer
`.spit()` method converts PhishLLM logs into SSE messages and send it to a user session. Must specify event type:
- `AnnouncerEvent.PROMPT`: this message is a prompt for PhishLLM
- `AnnouncerEvent.RESPONSE`: this message is a response from PhishLLM
- `AnnouncerEvent.SUCCESS`: PhishLLM has finished analysing, terminates user session
- `AnnouncerEvent.FAIL`: something went wrong, terminates user session

## Requirements
```txt
flask_cors
flask_session
gevent
apscheduler
```

## Usage
Start the server
```
python -m server.server
```