"""Config for model dags"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path)


class Config:
    VAR_STT_MODEL_ML_NAME = "STT_MODEL_ML_NAME"
    VAR_STT_LANGUAGE = "STT_DEFAULT_LANGUAGE"
    VAR_STT_PROJECT_ID = "STT_PROJECT_ID"
    VAR_STT_LOCAL_DIR_PATH = "STT_LOCAL_DIR_PATH"
    VAR_STT_ROOT_DIR = "STT_ROOT_DIR"

    DEMO_DIR_NAME = "default"
    DEMO_FILE_NAME = "demo_case.wav"
    DEPENDENT_DAG = "llm_jira_vllm_store_issues_and_predict"
    SUMMARY_JIRA_ISSUE = "Issue by STT"

    FILE_FORMATS = ["wav", "mp3"]

    STT_BASE_URL_TEMPLATE = os.environ.get("BASE_URL_TEMPLATE")

    JIRA_SD_EMAIL = os.environ.get("JIRA_SD_EMAIL")
    JIRA_SD_API_KEY = os.environ.get("JIRA_SD_API_KEY")
    JIRA_PROJECT_NAME = os.environ.get("JIRA_LLM_PROJECT_NAME")
    JQL_STT_DATA = 'project = %s AND "Description" IS NOT EMPTY AND "Summary" ~ "%s"' % (
        JIRA_PROJECT_NAME,
        SUMMARY_JIRA_ISSUE,
    )

    PATTERN_FILE_FORMAT = re.compile(r'\.([^.\\/:*?"<>|\r\n]+)$')
