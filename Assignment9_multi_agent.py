"""
Multi-Agent Research Assistant & MCP Server
===========================================

Assignment requirements:
- LangGraph Supervisor Agent
- Research Agent
- Analysis Agent
- Research knowledge-base tool
- Analysis comparison tool accepting two text snippets
- Supervisor routes requests to the correct worker
- One test requiring both workers
- FastMCP server with 2 tools
- MCP client demonstrating both tools

Install:
    pip install -U langgraph langchain langchain-core langchain-ollama fastmcp

Ollama:
    ollama serve
    ollama pull qwen3:0.6b

Run Multi-Agent demo:
    python multi_agent_research_mcp_updated.py agent

Run MCP server:
    python multi_agent_research_mcp_updated.py server

Run MCP client:
    python multi_agent_research_mcp_updated.py client
"""

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool


# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = "qwen3:0.6b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ============================================================
# Ollama Model
# ============================================================

model = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
)


# ============================================================
# Mock Knowledge Base
# ============================================================

KNOWLEDGE_BASE = {
    "artificial intelligence": """
Artificial Intelligence (AI) is a field of computer science
focused on creating systems that can perform tasks that
normally require human intelligence.

AI applications include natural language processing,
computer vision, recommendation systems, robotics,
and autonomous vehicles.

Machine learning is a major part of modern AI. Machine
learning systems learn patterns from data instead of being
explicitly programmed for every situation.
""",

    "machine learning": """
Machine Learning (ML) is a subset of Artificial Intelligence.
It allows computers to learn patterns from data and use those
patterns to make predictions or decisions.

Common types of machine learning include supervised learning,
unsupervised learning, and reinforcement learning.

Machine learning is used in recommendation systems,
fraud detection, image recognition, and predictive analytics.
""",

    "deep learning": """
Deep Learning is a subset of machine learning that uses
artificial neural networks with multiple layers.

Deep learning is particularly effective for large and complex
datasets. It is widely used for computer vision, speech
recognition, natural language processing, and generative AI.
""",
}


# ============================================================
# Research Tool
# ============================================================

@tool
def retrieve_information(topic: str) -> str:
    """Retrieve factual information about a topic from the knowledge base."""

    topic = topic.lower().strip()

    for key, information in KNOWLEDGE_BASE.items():
        if key in topic or topic in key:
            return information.strip()

    return f"No information found for: {topic}"


# ============================================================
# Research Agent
# ============================================================

research_agent = create_agent(
    model=model,
    tools=[retrieve_information],
    system_prompt="""
You are the Research Agent.

Your expertise is retrieving factual information from
the provided knowledge base.

Always use retrieve_information when the user asks
about a topic.

Return relevant facts clearly and concisely.
Do not perform detailed comparisons.
""",
)


# ============================================================
# Analysis Tool
# ============================================================

@tool
def compare_information(text1: str, text2: str) -> str:
    """Compare two text snippets and return a structured comparison."""

    prompt = f"""
    Compare the following two pieces of information.
    
    TEXT 1:
    {text1}
    
    TEXT 2:
    {text2}
    
    Return exactly this structure:
    
    Similarities:
    - point 1
    - point 2
    
    Differences:
    - point 1
    - point 2
    
    Conclusion:
    short conclusion
    """

    response = model.invoke(prompt)

    return response.content


# ============================================================
# Analysis Agent
# ============================================================

analysis_agent = create_agent(
    model=model,
    tools=[compare_information],
    system_prompt="""
    You are the Analysis Agent.
    
    Your job is to compare two pieces of information.
    
    When two text snippets are provided, use the
    compare_information tool.
    
    Return the comparison clearly with:
    1. Similarities
    2. Differences
    3. Conclusion
    """,
    )


# ============================================================
# Research Worker Tool
# ============================================================

@tool
def research_worker(question: str) -> str:
    """Ask the Research Agent to retrieve information."""

    response = research_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    return response["messages"][-1].content


# ============================================================
# Analysis Worker Tool
# ============================================================

@tool
def analysis_worker(question: str) -> str:
    """
    Ask the Analysis Agent to compare two topics
    from the knowledge base.
    """

    question_lower = question.lower()

    # Identify two supported topics.
    if (
        "artificial intelligence" in question_lower
        and "machine learning" in question_lower
    ):
        text1 = KNOWLEDGE_BASE["artificial intelligence"]
        text2 = KNOWLEDGE_BASE["machine learning"]

    elif (
        "machine learning" in question_lower
        and "deep learning" in question_lower
    ):
        text1 = KNOWLEDGE_BASE["machine learning"]
        text2 = KNOWLEDGE_BASE["deep learning"]

    else:
        return (
            "Analysis requires two topics available "
            "in the knowledge base."
        )

    analysis_prompt = f"""
Compare the following two pieces of information.

TEXT 1:
{text1}

TEXT 2:
{text2}

Use your comparison tool and provide:
1. Similarities
2. Differences
3. Conclusion
"""

    response = analysis_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": analysis_prompt,
                }
            ]
        }
    )

    return response["messages"][-1].content


# ============================================================
# Supervisor Agent
# ============================================================

supervisor = create_agent(
    model=model,
    tools=[
        research_worker,
        analysis_worker,
    ],
    system_prompt="""
You are the Supervisor Agent.

You have two specialist workers.

RESEARCH WORKER:
Use this worker when the user wants information
about a topic.

Examples:
- What is X?
- Explain X.
- Tell me about X.

ANALYSIS WORKER:
Use this worker when the user wants to compare
or analyse two topics.

Examples:
- Compare X and Y.
- What is the difference between X and Y?
- Analyse X and Y.

IMPORTANT:

If the user asks to first research information
and then compare it, use both workers.

For a research-only request:
1. Call research_worker.
2. Return its result.

For a comparison request:
1. Call analysis_worker.
2. Return its result.

For a request requiring research followed by comparison:
1. Call research_worker.
2. Then call analysis_worker.
3. Return a clear final answer.

Always use the appropriate worker instead of
performing specialist work yourself.
""",
)


# ============================================================
# Multi-Agent Demo
# ============================================================

def run_agent_demo():

    print("\n" + "=" * 70)
    print("MULTI-AGENT RESEARCH ASSISTANT")
    print("=" * 70)

    # --------------------------------------------------------
    # TEST 1 - Research Agent
    # --------------------------------------------------------

    question = "What is artificial intelligence?"

    print("\n" + "-" * 70)
    print("TEST 1 - RESEARCH AGENT")
    print("-" * 70)
    print("Question:")
    print(question)

    response = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print("\nAnswer:")
    print(response["messages"][-1].content)

    # --------------------------------------------------------
    # TEST 2 - Analysis Agent
    # --------------------------------------------------------

    question = """
Compare artificial intelligence and machine learning.
Explain their similarities and differences.
"""

    print("\n" + "-" * 70)
    print("TEST 2 - ANALYSIS AGENT")
    print("-" * 70)
    print("Question:")
    print(question)

    response = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print("\nAnswer:")
    print(response["messages"][-1].content)

    # --------------------------------------------------------
    # TEST 3 - Both Workers
    # --------------------------------------------------------

    question = """
Research artificial intelligence and machine learning
from the knowledge base, then compare them and explain
how machine learning is related to artificial intelligence.
"""

    print("\n" + "-" * 70)
    print("TEST 3 - BOTH WORKERS")
    print("-" * 70)
    print("Question:")
    print(question)

    response = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print("\nAnswer:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    run_agent_demo()

