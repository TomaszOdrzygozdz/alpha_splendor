"""Tests for alpacka.agents.deterministic_mcts."""

from alpacka import agents
from alpacka import envs
from alpacka import testing
from alpacka.networks.keras import _make_inputs, _make_output_heads
import keras


def test_integration_with_cartpole():
    env = envs.CartPole()
    agent = agents.DeterministicMCTSAgent(n_passes=2)
    network_sig = agent.network_signature(
        env.observation_space, env.action_space
    )
    episode = testing.run_with_dummy_network_prediction(
        agent.solve(env), network_sig)
    assert episode.transition_batch.observation.shape[0]  # pylint: disable=no-member



#TODO finish this test
def test_networks_with_multiple_inputs():
    def multi_input_network(network_signature, output_activation, output_zero_init):
        inputs = _make_inputs(network_signature.input)
        x = keras.layers.Concatenate(inputs)
        outputs = _make_output_heads(
            x, network_signature.output, output_activation, output_zero_init
        )

        return keras.Model(inputs=inputs, outputs=outputs)

