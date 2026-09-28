import ast
import math
import os
from datetime import datetime
import json

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from groq import Groq
import httpx
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

# ============== CONFIGURATION ==============


conversation_memory = {}


def _safe_calculate(expression):
    """Evaluate only basic arithmetic, using an explicit AST allowlist."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError("Expression must be a non-empty string of at most 200 characters")

    binary_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    unary_ops = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large")
            result = binary_ops[type(node.op)](left, right)
            if not math.isfinite(result) or abs(result) > 1e100:
                raise ValueError("Calculation result is out of range")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](evaluate(node.operand))
        raise ValueError("Only basic arithmetic is allowed")

    return evaluate(ast.parse(expression, mode="eval"))

# ============== MCP TOOLS ==============

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_info",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def execute_tool(tool_name, tool_args):
    """Execute MCP tool"""
    try:
        if tool_name == "get_weather":
            city = tool_args.get("city", "")
            return {
                "city": city,
                "temperature": "22°C",
                "condition": "Sunny",
                "humidity": "65%"
            }
        
        elif tool_name == "calculate":
            expression = tool_args.get("expression", "")
            result = _safe_calculate(expression)
            return {"result": result, "expression": expression}
        
        elif tool_name == "search_info":
            query = tool_args.get("query", "")
            return {
                "query": query,
                "results": f"Information about {query}: This is a placeholder result."
            }
        
        return {"error": "Unknown tool"}
        
    except Exception as e:
        return {"error": str(e)}


# ============== GROQ + MCP FUNCTION ==============

def get_llm_response_with_mcp(user_message, sender_number=None):
    """Get response from Groq with MCP tools"""
    try:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured")
        client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client(proxy=None))
        
        # Get conversation history
        if sender_number and sender_number in conversation_memory:
            messages = list(conversation_memory[sender_number])
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful WhatsApp assistant with access to tools. Keep responses concise."
                }
            ]
        
        messages.append({"role": "user", "content": user_message})
        
        # Initial call with tools
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # If no tool calls, return response
        if not tool_calls:
            bot_response = response_message.content
            messages.append({"role": "assistant", "content": bot_response})
        else:
            # Add assistant message with tool calls
            messages.append(response_message.model_dump(exclude_none=True))
            
            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                app.logger.info("Executing tool: %s", function_name)
                
                # Execute tool
                function_response = execute_tool(function_name, function_args)
                
                # Add tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
            
            # Get final response with tool results
            final_response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=1024
            )
            
            bot_response = final_response.choices[0].message.content
            messages.append({"role": "assistant", "content": bot_response})
        
        # Save conversation (keep last 10 messages)
        if sender_number:
            if len(messages) > 11:
                messages = [messages[0]] + messages[-10:]
            conversation_memory[sender_number] = messages
        
        return bot_response
        
    except Exception as e:
        app.logger.exception("Groq response failed")
        return "Sorry, I couldn't process that message right now. Please try again."


def clear_conversation(sender_number):
    """Clear conversation history"""
    if sender_number in conversation_memory:
        del conversation_memory[sender_number]


# ============== WHATSAPP FUNCTIONS ==============

def send_whatsapp_message(to_number, message):
    """Send WhatsApp message using Twilio"""
    try:
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
            raise RuntimeError("Twilio WhatsApp configuration is incomplete")
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=to_number
        )
        return msg.sid
    except Exception as e:
        app.logger.exception("Twilio send failed")
        return None


# ============== WEBHOOK ENDPOINTS ==============

@app.route("/webhook/twilio", methods=["POST"])
def twilio_webhook():
    """Webhook for Twilio WhatsApp"""
    try:
        incoming_msg = request.values.get("Body", "").strip()
        sender_number = request.values.get("From", "")
        sender_name = request.values.get("ProfileName", "User")

        if not incoming_msg or not sender_number:
            resp = MessagingResponse()
            resp.message("Please send a message to get started.")
            return str(resp)
        
        app.logger.info("Incoming WhatsApp message from %s", sender_name)
        
        # Check commands
        if incoming_msg.lower() in ["clear", "reset", "new chat"]:
            clear_conversation(sender_number)
            bot_response = "🔄 Conversation cleared!"
        else:
            # Get LLM response with MCP
            bot_response = get_llm_response_with_mcp(incoming_msg, sender_number)
        
        # Send response
        resp = MessagingResponse()
        resp.message(bot_response)
        
        return str(resp)
        
    except Exception as e:
        app.logger.exception("Twilio webhook failed")
        resp = MessagingResponse()
        resp.message("Sorry, error occurred.")
        return str(resp)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": GROQ_MODEL,
        "active_conversations": len(conversation_memory),
        "mcp_enabled": True
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Bot statistics"""
    return jsonify({
        "active_conversations": len(conversation_memory),
        "model": GROQ_MODEL,
        "users": list(conversation_memory.keys())
    })


@app.route("/", methods=["GET"])
def home():
    """Home page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WhatsApp MCP Bot</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #25D366; }
            .status { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🤖 WhatsApp MCP Bot with Groq</h1>
        <div class="status">
            <strong>Status:</strong> ✅ Active<br>
            <strong>Model:</strong> """ + GROQ_MODEL + """<br>
            <strong>MCP Tools:</strong> Weather, Calculator, Search<br>
            <strong>Active Conversations:</strong> """ + str(len(conversation_memory)) + """
        </div>
        <h2>Available MCP Tools:</h2>
        <ul>
            <li><strong>Weather:</strong> "What's the weather in London?"</li>
            <li><strong>Calculator:</strong> "Calculate 25 * 4 + 10"</li>
            <li><strong>Search:</strong> "Search for Python tutorials"</li>
        </ul>
        <h2>Commands:</h2>
        <ul>
            <li><code>clear</code> / <code>reset</code> - Clear conversation</li>
        </ul>
    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WhatsApp Bot with Groq and tools")
    print("=" * 60)
    print(f"Model: {GROQ_MODEL}")
    print("Server: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
