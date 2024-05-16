"""
1. load the data from xlsx
2. get unique company name
3. for each company, get all job description
4. call chatgpt4 to get the most reference jobs
4. write to another dataframe
"""

import logging
from typing import Dict, List

import openpyxl
import pandas as pd
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

from configs.log_config import setup_logging
from prompt.job_requirement import create_job_requirement_chain
from prompt.resume_job_matcher import Job, create_job_matcher_chain, get_chain_input

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


def load_excel_data(file_path, sheet_name=None):
    """Load Excel data and return it as a pandas DataFrame."""
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active
    data = ws.values
    columns = next(data)[0:]  # Extract the first row as column names
    return pd.DataFrame(data, columns=columns)


def group_job_descriptions_by_company(
    df,
    job_title_col="Job_title",
    company_col="Company",
    job_req_col="Job_requirement",
    link_col="Job_link",
) -> Dict[str, List[Job]]:
    """Group job descriptions by company and return a dictionary."""
    company_jobs = {}
    for idx, row in df.iterrows():
        if row[company_col] not in company_jobs:
            company_jobs[row[company_col]] = []
        company_jobs[row[company_col]].append(
            Job(idx, row[job_title_col], row[job_req_col], link=row[link_col])
        )
    return company_jobs


def write_to_excel(output_data, file_path):
    """Write output data to an Excel file using pandas."""
    output_df = pd.DataFrame(output_data)
    output_df.to_excel(file_path, index=False)


def main():
    # Constants and input/output paths
    input_path = "linkedin_job_requirement.xlsx"
    output_path = "output_data.xlsx"

    setup_logging()

    # Load Excel data into DataFrame
    df = load_excel_data(input_path)

    # init llm to generate result
    llm = ChatOpenAI(request_timeout=120, default_headers={"Host": "api.openai.com"})

    # OpenAI chain to extract job requirement
    job_req_chain = create_job_requirement_chain(llm)

    # Extract job requirement from description
    # with get_openai_callback() as cb:
    #     for idx, jd in enumerate(df["Job_description"]):
    #         df.at[idx, "Job_requirement"] = job_req_chain.invoke(jd)
    #     logger.info(cb)

    # iterate companies and find the most matched jobs
    company_jobs_map = group_job_descriptions_by_company(df)

    # get resume
    resume = open("resume.txt", encoding="utf-8").read()

    job_matcher_chain = create_job_matcher_chain(llm)
    companies = []
    descriptions = []
    with get_openai_callback() as cb:
        for company, jobs in company_jobs_map.items():
            jobs_and_resume = get_chain_input(jobs, resume)
            result = job_matcher_chain.invoke(jobs_and_resume)
            companies.append(company)
            descriptions.append(result)
            break
        logger.info(cb)

    # Write results to an Excel file
    result_df = pd.DataFrame({"company": companies, "description": descriptions})
    result_df.to_excel(output_path, index=False)

    print(f"Processing completed and data written to {output_path}")


if __name__ == "__main__":
    main()
