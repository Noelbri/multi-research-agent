import os
from typing import Annotated, Dict, List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
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


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

tavily_tool = TavilySearchResults(max_results=5)

#This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return{
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

members = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    "following workers: {members}. Given the following user request,"
    "respond with the worker to act next. Each worker will perform a"
    "task and respond with their results and status. When finished,"
    "respond with FINISH."
)

#Our team supervisor is an LLM node. It just picks the next agent to process
#and decides when the work is completed
options = ["FINISHED"] + members

class routeResponse(BaseModel):
    next:Literal[*options]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagePlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            "Or should we FINISH? Select one of: {options}",
        )
    ]
).partial(options=str(options), members=", ".join(members))

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)

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
    workflow.add_edge(member, "supervisor")

#The supervisor populates the "next" field in the graph state
#which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
#finally add entrypoint
workflow.add_edge(START, "Supervisor")

graph = workflow.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    #Initialize the graph state with the user input
    state = {"messages": [HumanMessage(content=user_input)]}
    #Run the graph
    result = graph.run(state)
    #Print the final result
    print("Supervisor:", result["messages"][-1].content)

    