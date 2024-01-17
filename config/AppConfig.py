import os
from pathlib import Path



from utils.config_utils import EnvInit
from utils.logger_utils import logger

env=EnvInit()
logger.debug(f'env repo{env.REPO}')
REPO_PATH = Path(os.path.abspath(env.REPO or '../repo'))
ORCHESTRATOR = f"{env.GATEWAY}/routes/orchestrator/"


HF_TOKEN = env.HF_TOKEN
HF_TOKEN = None if HF_TOKEN == "<insert your HF token here>" else HF_TOKEN