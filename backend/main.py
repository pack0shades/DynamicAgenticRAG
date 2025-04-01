import os
import openai
from typing import List
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from concurrent.futures import ThreadPoolExecutor
from src.reranker import jina_reranker
from src.router import get_routed_agent
from PathwayVectorStore.vectorRetriever import VectorStoreRetriever
from PathwayVectorStore.vectorRetrieverHybrid import VectorStoreRetrieverHybrid
from src.dynamic_crew import get_dynamic_agents
from src.config import (
    META_AGENT_PROMPT,
    META_AGENT_TASK,
    MARKDOWN_TASK,
    QUE_ANS_AGENT_PROMPT,
    QUE_ANS_AGENT_TASK,
    ALLOW_DELEGATION,
    MODEL,
    LOG_PATH,
    HOST_NAME,
    DENSE_PORT,
    SPARSE_PORT
)
from src.crewai_reflection import reflection
from src.crewai_multi import get_multi_agents
from src.guardrail import ( 
    guardrails,
    query_refinment
)
from crewai import (
    Agent,
    Task,
    Crew,
    Process
)
from typing import List
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up vector retrievers
vector_retriever_dense = VectorStoreRetriever(HOST_NAME, port=DENSE_PORT)
vector_retriever_sparse = VectorStoreRetriever(HOST_NAME, port=SPARSE_PORT)
vector_retriever_hyprid = VectorStoreRetrieverHybrid(
    vector_retriever_dense,
    vector_retriever_sparse
)

def get_context(
    query: str,
    topk: int,
    reranker: bool=False,
    method: str="cr"
)-> List[str]:

    # Get context from vector retrievers
    if method == "cr" or method == "hs":
        retrieved_docs = vector_retriever_hyprid.rank_fusion(
            query=query,
            top_k=topk*6,  # 6 is the number of vector retrievers
        )
    elif method == None:
        retrieved_docs = vector_retriever_dense.get_context(
            query=query,
            top_k=topk*6
        )
    else:
        raise ValueError(f"Invalid method: {method}")
    
    logger.trace(f"Retrieved {len(retrieved_docs)} documents")
    
    # Apply reranker
    if reranker == True:
        retrieved_docs = jina_reranker(
            query=query,
            documents=retrieved_docs,
            topk=topk
        )
    else:
        retrieved_docs = retrieved_docs[:topk]

    logger.debug(f"Retrieved docs:\n {retrieved_docs}")
    logger.debug(f"Retrieved {len(retrieved_docs)} documents after reranking")

    context = " ".join(retrieved_docs)
    logger.debug(f"Retrieved context:\n {context}")
    return context


def pipeline(
    query: str,
    topk: int,
    reranker: bool=True,
    method: str="cr", 
    agent_type: str = "dynamic",
    use_reflection: bool=True,
    n_reflection: int=2, 
    use_router: bool=True
)-> str:
    
    # Check if agent type is valid
    if agent_type!="dynamic" and agent_type!="multi":
        raise ValueError(f"Invalid agent type: {agent_type} --> choose one of 'dynamic' or 'multi'")
    
    # Apply initial guardrails
    passed = guardrails(query)

    logger.info(f"Query after guardrails: {query}")
    if passed == False:
        return "Query has toxic language"
    
    # Apply query refinment
    query = query_refinment(query)

    # Parallelize context retrieval for efficiency
    with ThreadPoolExecutor() as executor:
        context_future = executor.submit(
            get_context,
            query=query,
            topk=topk,
            reranker=reranker,
            method=method
        )
        
        # Chain `get_dynamic_agents` to the result of `context_future`
        if agent_type == "dynamic":
            # Submit the document context retrieval
            doc_context_future = executor.submit(
                vector_retriever_dense.get_doc_text,
            )
            doc_context = doc_context_future.result()
            logger.debug(f"type of doc_context: {type(doc_context)}")
            # Chain `get_dynamic_agents` to the result of `doc_context_future`
            dynamic_agents_future = executor.submit(
                get_dynamic_agents,
                document=doc_context
            )
            
        else:
            dynamic_agent_list = []
            dynamic_tasks_list = []
            # if pipeline is not dynamic get predefined agents
            static_agent_list, static_tasks_list = get_multi_agents()
        
        context = context_future.result()

    # A general QA agent
    que_ans_agent = Agent(
        role=QUE_ANS_AGENT_PROMPT["role"],
        goal=QUE_ANS_AGENT_PROMPT["goal"],
        backstory=QUE_ANS_AGENT_PROMPT["backstory"],
        verbose=QUE_ANS_AGENT_PROMPT["verbose"],
        allow_delegation=ALLOW_DELEGATION,
        llm=MODEL
    )

    que_ans_task_description = QUE_ANS_AGENT_TASK["description"].format(query, context)
    que_ans_task = Task(
        description=que_ans_task_description,
        expected_output=QUE_ANS_AGENT_TASK["expected_output"],
        agent=que_ans_agent,
        async_execution=True
    )

    static_agent_list, static_tasks_list =[],[]

    if agent_type == "dynamic":
            dynamic_agent_list, dynamic_tasks_list = dynamic_agents_future.result()
            static_agent_list = [que_ans_agent]
            static_tasks_list = [que_ans_task]

    if use_router and len(dynamic_agent_list)>0:
        while True:
            try:
                dynamic_agent_list, dynamic_tasks_list = get_routed_agent(
                    agent_list=dynamic_agent_list,
                    task_list=dynamic_tasks_list,
                    query=query
                )
                break
            except:
                logger.error("Error in routing, trying again")
                continue
            
    
    agent_list = static_agent_list + dynamic_agent_list
    tasks_list = static_tasks_list + dynamic_tasks_list

    logger.debug(f"Agent list: {agent_list}")

    meta_agent= Agent(
            role= META_AGENT_PROMPT["role"],
            goal= META_AGENT_PROMPT["goal"],
            backstory= META_AGENT_PROMPT["backstory"],
            verbose=True,
            allow_delegation= META_AGENT_PROMPT["allow_delegation"]
        )

    meta_agent_task_description = META_AGENT_TASK["description"].format(que=query)

    meta_agent_task = Task(
            description=meta_agent_task_description,
            expected_output=META_AGENT_TASK["expected_output"],
            agent=meta_agent,
            context= tasks_list,
            async_execution=False
        )
    
    logger.debug(f"Meta agent task description: {meta_agent_task_description}")

    if use_reflection:
        response = reflection(
            query=query,
            context=context,
            agents=agent_list,
            agents_task=tasks_list,
            meta_agent=meta_agent,
            meta_agent_task=meta_agent_task,
            n=n_reflection)
        
    else:
        string_source = StringKnowledgeSource(
            content=context,
            metadata={"preference": "personal"}
        )
        meta_agent_task.expected_output = MARKDOWN_TASK
        initial_crew = Crew(
            agents= [*agent_list, meta_agent],
            tasks= [*tasks_list, meta_agent_task],
            process=Process.sequential,
            verbose=True,
            knowledge={
                "sources": [string_source],
                "collection_name": "crew_knowledge",
            },
            memory=False,
            output_log_file=LOG_PATH
        )

        inputs = {
            'que': query,
            'con': context
        }

        def sanitize_inputs(inputs: dict) -> dict:
            sanitized_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, str):
                    # Escape any curly braces in the string values
                    sanitized_inputs[key] = value.replace("{", "{{").replace("}", "}}")
                else:
                    sanitized_inputs[key] = value
            return sanitized_inputs

        inputs = sanitize_inputs(inputs) # escape curly braces in the inputs
        
        response = initial_crew.kickoff(inputs=inputs).raw
        logger.info(f"response without guardrail--{response}\nfor the query:\n{query}")

    passed = guardrails(response)

    if passed is True:
        logger.info (f"Final response after final guardrail:\n {response} \nfor the query:\n{query}")
        return response
    
    else:
        return "Sorry the response contains sensitive content. Please re-enter the query if the error still perssists check the document content."


def main():
    query = "what is BB84 protocol?"
    
    pipeline(
        query=query,
        topk=5,
        reranker=True,
        method="cr",
        agent_type="dynamic",
        use_reflection=False,
        n_reflection=4,
        use_router=True
    )

if __name__ == "__main__":
    main()
