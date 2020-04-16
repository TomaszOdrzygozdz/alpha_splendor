import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='splendor-base-v0',
    entry_point='splendor.envs.splendor:SplendorEnv'
)