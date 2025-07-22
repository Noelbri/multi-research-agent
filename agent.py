import os
from typing import Annotated, Dict, List, Optional
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

# Use the newer tavily package to avoid deprecation warning
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    # Fallback to the older import if new package not available
    from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

tavily_tool = TavilySearchResults(max_results=5, api_key=tavily_api_key)

#This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return{
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

members = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the "
    "following workers: {members}. Given the following user request, "
    "respond with the worker to act next. Each worker will perform a "
    "task and respond with their results and status. "
    "For tasks requiring both research and coding (like creating reports), "
    "first send to Researcher to gather information, then to Coder to "
    "create the final output. When finished, respond with FINISHED."
)

#Our team supervisor is an LLM node. It just picks the next agent to process
#and decides when the work is completed
options = ["FINISHED"] + members

class routeResponse(BaseModel):
    next: Literal["FINISHED", "Researcher", "Coder"]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            "Or should we FINISH? Select one of: {options}",
        )
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    response = supervisor_chain.invoke(state)
    return {"next": response.next}

#The agent state is the input to each node in the graph.
class AgentState(TypedDict):
    #The annotation tells the graph that new messages will always
    #be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    #The 'next' field indicates where to route to next 
    next: str
    
researcher_agent = create_react_agent(llm, tools=[tavily_tool])
researcher_node = functools.partial(agent_node, agent=researcher_agent, name="Researcher")

#NOTE: This performs arbitrary code execution, which can be unsafe
coder_agent = create_react_agent(llm, tools=[python_repl_tool])
coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("Supervisor", supervisor_agent)

for member in members:
    #We want our workers to ALWAYS "report back" to the supervisor when done 
    workflow.add_edge(member, "Supervisor")

#The supervisor populates the "next" field in the graph state
#which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISHED"] = END
workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
#finally add entrypoint
workflow.add_edge(START, "Supervisor")

graph = workflow.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    
    print("🤖 Multi-Agent System Processing...")
    #Initialize the graph state with the user input
    state = {"messages": [HumanMessage(content=user_input)]}
    
    #Run the graph and show progress
    try:
        result = graph.invoke(state)
        #Print the final result
        print("\n" + "="*50)
        print("📋 FINAL RESULT:")
        print("="*50)
        
        # Show all agent responses
        for i, message in enumerate(result["messages"]):
            if hasattr(message, 'name') and message.name:
                print(f"\n🎯 {message.name}:")
                print(f"{message.content}")
            elif i == 0:  # First message is user input
                print(f"\n👤 User: {message.content}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please try again or type 'exit' to quit.")
