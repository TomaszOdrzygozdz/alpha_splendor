import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='splendor-v0',
    entry_point='splendor.envs.splendor:SplendorEnv',
)

register(
    id='splendor-v1',
    entry_point='splendor.envs.splendor_wrapper:SplendorWrapperEnv',

)

register(
    id='splendor-deterministic-v0',
    entry_point='splendor.envs.splendor_deterministic:SplendorDeterministic',
)