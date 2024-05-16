import logging
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from configs.log_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class Job:
    idx: int
    position: str
    requirements: str
    link: str

    def to_template(self):
        return f"""\
{str(self.idx)}. **{self.position}:**
# requirements #
{self.requirements}
# job link #
{self.link}
"""


SYSTEM_PROMPT = """\
You are an assistant specialized in analyzing resumes to suggest the best job fit or identify gaps in qualifications.
"""

HUMAN_PROMPT = """\
Here are the job requirements for the positions at the company:

# Position #
{positions}

# My Profile #
{resume}

Identify the best matching position and format your response as:
[row_num]. **[position]:**
   - [job_requirements 1]
   - [job_requirements 2]
   - [job_requirements 3]
   - ...
   - [link].
[following_reason]
If no match fits perfectly, suggest how my qualifications could be improved.
"""


def get_template_by_prompt(chat_format: str, human_prompt: str) -> ChatPromptTemplate:
    """create the requirement extraction by prompt

    Args:
        chat_format (str): prompt string that ask llm to extract the requirement from jd

    Returns:
        ChatPromptTemplate: the template that allow to input the job description
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=chat_format),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]
    )


def parseOutput(ai_message: AIMessage) -> str:
    return ai_message.content


def create_job_matcher_chain(llm: LLM) -> Runnable:
    """create a template for getting the job requirement"""
    prompt = get_template_by_prompt(SYSTEM_PROMPT, HUMAN_PROMPT)
    return prompt | llm | parseOutput


def get_chain_input(jobs: List[Job], resume: str) -> Dict[str, str]:
    job_strs = [job.to_template() for job in jobs]
    return {"positions": "\n###########\n".join(job_strs), "resume": resume}
