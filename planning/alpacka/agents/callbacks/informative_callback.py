import gin

from alpacka import agents

@gin.configurable
class InformativeCallback(agents.AgentCallback):
    """Base class for agent callbacks."""

    # Events for all OnlineAgents.

    def on_episode_begin(self, env, observation, epoch):
        """Called in the beginning of a new episode."""

    def on_episode_end(self):
        """Called in the end of an episode."""

    def on_real_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the real environment."""

    def on_pass_begin(self):
        """Called in the beginning of every planning pass."""

    def on_pass_end(self):
        """Called in the end of every planning pass."""

    def on_model_step(self, agent_info, action, observation, reward, done):
        """Called after every step in the model."""