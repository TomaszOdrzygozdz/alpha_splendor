"""TraceX entrypoint."""

import functools
import sys

import flask
import numpy as np

from alpacka import data as alpacka_data
from alpacka import tracing


app = flask.Flask(__name__, static_url_path='', static_folder='static')
rendered_trajectory = None


@app.route('/')
def index():
    return app.send_static_file('index.html')


def render_trajectory(trajectory):
    states = [trajectory.init_state] + [
        transition.to_state for transition in trajectory.transitions
    ]
    state_passes = [
        transition.passes for transition in trajectory.transitions
    ] + [[]]

    def to_primitive(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        else:
            return x

    def to_primitive_dict(obj, keys):
        if obj is not None:
            return {
                key: alpacka_data.nested_map(to_primitive, getattr(obj, key))
                for key in keys
            }
        else:
            return {}

    render_state = functools.partial(
        to_primitive_dict, keys=('terminal', 'state_info')
    )
    render_transition = functools.partial(
        to_primitive_dict, keys=('action', 'reward')
    )

    def render_pass(pass_, init):
        transition = pass_[0]
        data = {
            'type': 'model_init' if init else 'model',
            'action': transition.action,
            'reward': transition.reward,
            **render_state(transition.to_state),
            **render_transition(transition),
        }
        if len(pass_) > 1:
            data['children'] = [render_pass(pass_[1:], False)]
        return data

    return {
        'type': 'root',
        'children': [
            {
                'type': 'real',
                'children': [
                    render_pass(pass_, True) for pass_ in passes if pass_
                ],
                **render_state(state),
                **render_transition(transition),
            }
            for (state, passes, transition) in zip(
                states, state_passes, [None] + trajectory.transitions
            )
        ]
    }


@app.route('/data')
def data():
    return flask.jsonify(rendered_trajectory)


if __name__ == '__main__':
    trace = tracing.load(sys.argv[1])
    rendered_trajectory = render_trajectory(trace.trajectory)
    app.run()
