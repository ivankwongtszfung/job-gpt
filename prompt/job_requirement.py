import logging

from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """\
You will receive a job description from the user. Your task is to identify and extract the job requirements, organizing them into clear and concise bullet points. Follow these guidelines:

1. **Required Qualifications**: List the mandatory skills, qualifications, and experiences essential for the job.
2. **Preferred Qualifications**: List any additional skills, qualifications, and experiences that enhance a candidate's suitability but are not strictly required.

Structure the response like this:

**Required Qualifications:**
- [Key Requirement 1]
- [Key Requirement 2]
- ...

**Preferred Qualifications:**
- [Key Requirement 1]
- [Key Requirement 2]
- ...

Respond to the job description the user provides with this structured information.
"""
SELF_PROMPT_FORMAT = """\
I will provide a job description. analyze and extract all the job requirements in a structured bullet point list. Ensure each requirement is clear, concise, and accurately reflects the necessary qualifications, skills, and experience. Be sure to identify specific technical competencies, certifications, educational qualifications, and any other relevant attributes or criteria mentioned. Make sure that the requirements are comprehensive and reflect the essential expectations for prospective candidates.

Structure the response like this:

**Required Qualifications:**
- [Key Requirement 1]
- [Key Requirement 2]
- ...

...
"""
TARGET_ORIENTED_FORMAT = """\
# Task #
Analyze the provided job description and extract all job requirements into a clear, structured bullet point list.

# Context #
You will be given a job description that details the roles, responsibilities, and qualifications needed for a specific position.

# Exemplars #

Minimum 5 years of relevant experience.
Proficient in Java and SQL.
Excellent communication skills and team leadership experience.
# Persona #
Act as a detail-oriented HR analyst who is adept at parsing complex job descriptions to identify and list essential qualifications and skills.

# Format #
The output should be in the format of bullet points for each requirement extracted from the job description.

# Tone #
The tone should be formal and professional, suitable for a corporate HR setting, emphasizing clarity and precision in communication.
"""
COSTAR_FORMAT = """\
# Context #
The system will receive a job description as input. This description outlines roles, responsibilities, and the qualifications needed for a specific position.

# Objective #
The system is tasked to automatically analyze the job description and extract all necessary job requirements, listing them in a structured bullet point format.

# Style #
The output should be clear and concise, making each requirement easy to understand and directly actionable.

# Tone #
Maintain a neutral and factual tone, as the output will be used in a professional HR setting.

# Audience #
The primary users of this system are HR professionals and hiring managers who require quick, accurate job requirement lists to aid in recruitment processes.

# Response #
The system should produce a bullet point list of job requirements extracted from the provided job description. This list should be directly usable in HR processes without further editing.
"""
FORMATS = [COSTAR_FORMAT, SELF_PROMPT_FORMAT, TARGET_ORIENTED_FORMAT, SYSTEM_PROMPT]
HUMAN_PROMPT = """\
# Job Description #
{job_description}
"""


def get_template_by_prompt(chat_format: str) -> ChatPromptTemplate:
    """create the requirement extraction by prompt

    Args:
        chat_format (str): prompt string that ask llm to extract the requirement from jd

    Returns:
        ChatPromptTemplate: the template that allow to input the job description
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=chat_format),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
        ]
    )


def get_job_requirement_by_self_prompt(job_description: str, llm: LLM) -> str:
    """extract job_requirement by self_prompt created by gpt4

    Args:
        job_description (str): job description from linkedin
        llm (LLM): large language model

    Returns:
        str: job requirement generated by llm
    """
    prompt = get_template_by_prompt(SELF_PROMPT_FORMAT)
    llm_chain = prompt | llm
    return llm_chain.invoke(job_description)


def parseOutput(ai_message: AIMessage) -> str:
    return ai_message.content


def create_job_requirement_chain(llm) -> Runnable:
    """create a template for getting the job requirement"""
    prompt = get_template_by_prompt(SELF_PROMPT_FORMAT)
    return prompt | llm | parseOutput
