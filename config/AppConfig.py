import os
from pathlib import Path



from utils.config_utils import EnvInit
from utils.logger_utils import logger

env=EnvInit()
logger.debug(f'env repo{env.REPO}')
REPO_PATH = Path(os.path.abspath(env.REPO or '../repo'))
ORCHESTRATOR = f"{env.GATEWAY}/routes/orchestrator/"


