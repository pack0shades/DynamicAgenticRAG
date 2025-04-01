from crewai import Agent, Task, Crew, Process
from typing import List
from loguru import logger 
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from .config import (
    CRITIQUE_AGENT_PROMPT,
    CRITIQUE_AGENT_TASK,
    DYNAMIC_AGENT_TASK,
    META_AGENT_PROMPT,
    META_AGENT_TASK,
    MARKDOWN_TASK,
    MODEL,
    ALLOW_DELEGATION,
    LOG_PATH
)
from dotenv import load_dotenv
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

def reflection(
    query:str,
    context:str,
    agents:List[Agent],
    agents_task:List[Task],
    meta_agent:Agent,
    meta_agent_task:Task,
    n:int=2
)-> str:
    
    string_source = StringKnowledgeSource(
        content=context,
        metadata={"preference": "personal"}
    )
    
    initial_crew = Crew(
        agents=[*agents,
                meta_agent
                ],

        tasks=[*agents_task,
            meta_agent_task],

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

    inputs = sanitize_inputs(inputs)  # escape curly braces in the inputs

    results_inital = initial_crew.kickoff(inputs=inputs).raw

    logger.info(f"initial crew response in reflection-- {results_inital}")

    # critique agent which critiques the responses given by other agents and sends back to them for improving their responses during reflection.
    critique_agent= Agent(
        role= CRITIQUE_AGENT_PROMPT["role"],
        goal= CRITIQUE_AGENT_PROMPT["goal"],
        backstory= CRITIQUE_AGENT_PROMPT["backstory"],
        verbose=True,
        allow_delegation= ALLOW_DELEGATION,
        llm = MODEL
    )

    critique_task_description = CRITIQUE_AGENT_TASK["description"].format(que=query, con=context)
    critique_task = Task(
        description=critique_task_description,
        expected_output=CRITIQUE_AGENT_TASK["expected_output"],
        agent=critique_agent,
        context=[meta_agent_task],
        async_execution=False,
        output_log_file=LOG_PATH
    )

    dynamic_task_list = []
    # dynamic agents which improve the responses given by other agents
    dynamic_task_description = DYNAMIC_AGENT_TASK["description"].format(que=query, con=context)
    for i in range(len(agents)): # creating tasks for dynamic agents
        dynamic_agent_task = Task(
            description=dynamic_task_description,
            expected_output=DYNAMIC_AGENT_TASK["expected_output"],
            agent=agents[i],
            context=[critique_task],
            async_execution=True,
        )
        dynamic_task_list.append(dynamic_agent_task)


    final_agent= Agent(
            role= META_AGENT_PROMPT["role"],
            goal= META_AGENT_PROMPT["goal"],
            backstory= META_AGENT_PROMPT["backstory"],
            verbose=True,
            allow_delegation= ALLOW_DELEGATION,
            llm = MODEL
        )

    meta_agent_task_description = META_AGENT_TASK["description"].format(que=query)
    # final agent which aggregates the responses given by other agents and gives final response of pipeline
    final_agent_task = Task(
            description=meta_agent_task_description,
            expected_output=META_AGENT_TASK["expected_output"],
            agent=final_agent,
            context= dynamic_task_list,
            async_execution=False
        )
    
    logger.info(f"new tasks and final agent formed---")

    # reflection crew
    reflection_crew = Crew(
        agents=[critique_agent,
                *agents,
                final_agent],
        
        tasks=[critique_task,
            dynamic_agent_task,
            final_agent_task],
        
        process=Process.sequential,
        verbose=True,
        knowledge={
            "sources": [string_source],
            "collection_name": "crew_knowledge",
        },
        memory=False,
        output_log_file=LOG_PATH
    )

    logger.info(f"reflection crew formed---")

    inputs = {
        'question': query,
        'responses': results_inital
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
    ans=reflection_crew.kickoff(inputs=inputs).raw

    inputs['responses'] = ans

    logger.info(f"reflection at n=1--{inputs['responses']}")

    critique_task.context = [final_agent_task]
    for i in range(len(dynamic_task_list)):
        dynamic_task_list[i].context = [critique_task]
        final_agent_task.context = dynamic_task_list

    for _ in range(n-1):
        if _ == n-2:
            final_agent_task.expected_output = MARKDOWN_TASK
        ans = reflection_crew.kickoff(inputs=inputs).raw
        inputs['responses'] = ans
        logger.info(f"reflection at n={_+2}--{inputs['responses']}")

    logger.info(f"here is the final output from reflection crew:---{inputs['responses']}")
    return inputs['responses']
