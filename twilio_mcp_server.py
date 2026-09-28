import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from twilio.rest import Client

# Twilio credentials from environment variables
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID") or os.getenv("ACCOUNT_SID")
API_KEY = os.getenv("TWILIO_API_KEY") or os.getenv("API_KEY")
API_SECRET = os.getenv("TWILIO_API_SECRET") or os.getenv("API_SECRET")
PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER") or os.getenv("PHONE_NUMBER") or os.getenv("TWILIO_WHATSAPP_NUMBER")

if not all([ACCOUNT_SID, API_KEY, API_SECRET, PHONE_NUMBER]):
    raise ValueError("Missing required Twilio environment variables")

# Initialize Twilio client
twilio_client = Client(API_KEY, API_SECRET, ACCOUNT_SID)

# Create MCP server
server = Server("twilio-mcp-server")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="send_sms",
            description="Send an SMS message via Twilio",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number (E.164 format)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message body to send",
                    },
                },
                "required": ["to", "message"],
            },
        ),
        Tool(
            name="make_call",
            description="Make a phone call via Twilio",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number (E.164 format)",
                    },
                    "twiml_url": {
                        "type": "string",
                        "description": "URL of TwiML instructions for the call",
                    },
                },
                "required": ["to", "twiml_url"],
            },
        ),
        Tool(
            name="get_messages",
            description="Retrieve recent SMS messages",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Number of messages to retrieve (default: 20)",
                    }
                },
            },
        ),
        Tool(
            name="get_calls",
            description="Retrieve recent call logs",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Number of calls to retrieve (default: 20)",
                    }
                },
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    try:
        arguments = arguments or {}

        if name == "send_sms":
            if not all(k in arguments for k in ["to", "message"]):
                return [
                    TextContent(
                        type="text", text="Missing required arguments: to, message"
                    )
                ]
            message = twilio_client.messages.create(
                body=arguments["message"], from_=PHONE_NUMBER, to=arguments["to"]
            )
            return [
                TextContent(
                    type="text",
                    text=f"SMS sent successfully! SID: {message.sid}, Status: {message.status}",
                )
            ]

        elif name == "make_call":
            if not all(k in arguments for k in ["to", "twiml_url"]):
                return [
                    TextContent(
                        type="text", text="Missing required arguments: to, twiml_url"
                    )
                ]
            call = twilio_client.calls.create(
                url=arguments["twiml_url"], from_=PHONE_NUMBER, to=arguments["to"]
            )
            return [
                TextContent(
                    type="text",
                    text=f"Call initiated! SID: {call.sid}, Status: {call.status}",
                )
            ]

        elif name == "get_messages":
            limit = max(1, min(int(arguments.get("limit", 20)), 100))
            messages = twilio_client.messages.list(limit=limit)
            msg_list = []
            for msg in messages:
                msg_list.append(
                    {
                        "sid": msg.sid,
                        "from": msg.from_,
                        "to": msg.to,
                        "body": msg.body,
                        "status": msg.status,
                        "date": str(msg.date_sent),
                    }
                )
            return [TextContent(type="text", text=json.dumps(msg_list, indent=2))]

        elif name == "get_calls":
            limit = max(1, min(int(arguments.get("limit", 20)), 100))
            calls = twilio_client.calls.list(limit=limit)
            call_list = []
            for call in calls:
                call_list.append(
                    {
                        "sid": call.sid,
                        "from": getattr(
                            call,
                            "from_",
                            call.from_formatted
                            if hasattr(call, "from_formatted")
                            else "unknown",
                        ),
                        "to": call.to,
                        "status": call.status,
                        "duration": call.duration,
                        "date": str(call.date_created),
                    }
                )
            return [TextContent(type="text", text=json.dumps(call_list, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="twilio-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
