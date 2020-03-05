"""TraceX entrypoint."""

import sys

import flask

from alpacka import tracing


app = flask.Flask(__name__, static_url_path='', static_folder='static')
trace = None


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/data')
def data():
    transitions = trace.trajectory.transitions
    states = [trace.trajectory.init_state] + [
        transition.to_state for transition in transitions
    ]
    state_passes = [transition.passes for transition in transitions] + [[]]
    actions_and_rewards = [(None, None)] + [
        (transition.action, transition.reward) for transition in transitions
    ]

    def render_pass(pass_, init):
        transition = pass_[0]
        data = {
            'type': 'model_init' if init else 'model',
            'action': transition.action,
            'reward': transition.reward,
            'terminal': transition.to_state.terminal,
        }
        if len(pass_) > 1:
            data['children'] = [render_pass(pass_[1:], False)]
        return data

    return flask.jsonify({
        'type': 'root',
        'children': [
            {
                'type': 'real',
                'action': action,
                'reward': reward,
                'terminal': state.terminal,
                'children': [
                    render_pass(pass_, True) for pass_ in passes if pass_
                ]
            }
            for (state, passes, (action, reward)) in zip(
                states, state_passes, actions_and_rewards
            )
        ]
    })


if __name__ == '__main__':
    trace = tracing.load(sys.argv[1])
    app.run()
