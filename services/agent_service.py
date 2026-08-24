import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from services.forecaster import predict_stock_price
from services.rag_service import search_financial_docs

load_dotenv()


@tool
def get_stock_forecast(symbol: str, days: int = 1):
    """Fetches future price forecast and trend prediction for a stock ticker."""
    return predict_stock_price(symbol, days)


@tool
def search_company_reports(query: str):
    """Searches financial statements, balance sheets, profit/loss and company reports."""
    return search_financial_docs(query)


llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)
tools = [get_stock_forecast, search_company_reports]
llm_with_tools = llm.bind_tools(tools)

session_histories = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]


def format_ai_response(content) -> str:
    if isinstance(content, list) and len(content) > 0:
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        raw_text = "\n".join(text_parts) if text_parts else str(content)
    else:
        raw_text = str(content)

    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return cleaned_text if cleaned_text else raw_text.strip()


def ask_financial_agent(user_prompt: str, session_id: str = "default_session", lang: str = "tr"):
    tool_map = {
        "get_stock_forecast": get_stock_forecast,
        "search_company_reports": search_company_reports
    }

    history = get_session_history(session_id)

    try:
        current_messages = list(history.messages) + [HumanMessage(content=user_prompt)]
        ai_msg = llm_with_tools.invoke(current_messages)

        final_reply = ""

        if ai_msg.tool_calls:
            context_data = []
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                if tool_name in tool_map:
                    selected_tool = tool_map[tool_name]
                    tool_output = selected_tool.invoke(tool_call["args"])
                    context_data.append(f"[{tool_name} Output]:\n{tool_output}")

            combined_context = "\n\n".join(context_data)

            if lang == "en":
                synthesis_prompt = f"""You are a Senior Financial Analyst and Investment Expert.

User Query:
{user_prompt}

Data Retrieved from Systems and Financial Documents:
{combined_context}

TASK:
Evaluate the conversation history and the newly retrieved financial data above. Provide a thorough, professional, and well-structured response in English using Markdown formatting and data tables where applicable."""
            else:
                synthesis_prompt = f"""Sen kıdemli bir Finansal Analist ve Yatırım Uzmanısın.

Kullanıcının Güncel Sorusu:
{user_prompt}

Sistemden ve Finansal Dokümanlardan Elde Edilen Güncel Veriler:
{combined_context}

GÖREV:
Geçmiş konuşmayı ve yukarıdaki yeni finansal verileri birlikte değerlendirerek net, profesyonel, sayısal verileri ve tabloları vurgulayan düzenli bir Türkçe Markdown formatında yanıt ver."""

            messages_for_synthesis = list(history.messages) + [HumanMessage(content=synthesis_prompt)]
            synthesis_res = llm.invoke(messages_for_synthesis)
            final_reply = format_ai_response(synthesis_res.content)
        else:
            final_reply = format_ai_response(ai_msg.content)

        history.add_user_message(user_prompt)
        history.add_ai_message(final_reply)

        return final_reply

    except Exception as e:
        return f"⚠️ Error: {str(e)}"