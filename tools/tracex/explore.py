"""TraceX entrypoint."""

import functools
import sys

import flask
import numpy as np

from alpacka import data as alpacka_data
from alpacka import tracing


app = flask.Flask(__name__, static_url_path='', static_folder='static')
rendered_trajectory = None
entities = None


@app.route('/')
def index():
    return app.send_static_file('index.html')


def render_trajectory(trajectory):
    entities = {}

    def add_entity(entity):
        entity['id'] = len(entities)
        entities[entity['id']] = entity
        return entity

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
        return add_entity(data)

    def show_children_if_nonempty(children):
        if children:
            return {'children': children}
        else:
            return {}

    root = add_entity({
        'type': 'root',
        'children': [
            add_entity({
                'type': 'real',
                **show_children_if_nonempty([
                    render_pass(pass_, True) for pass_ in passes if pass_
                ]),
                **render_state(state),
                **render_transition(transition),
            })
            for (state, passes, transition) in zip(
                states, state_passes, [None] + trajectory.transitions
            )
        ]
    })
    return (root, entities)


def lazify_entity(entity, depth, lazy_keys):
    if isinstance(entity, dict):
        entity = {
            key: (
                value if key not in lazy_keys
                else lazify_entity(value, depth, lazy_keys)
            )
            for (key, value) in entity.items()
        }
        if depth == 0:
            entity['stub'] = True
        return entity
    elif isinstance(entity, (list, tuple)):
        if depth == 0:
            return type(entity)()
        else:
            return type(entity)(
                lazify_entity(x, depth - 1, lazy_keys) for x in entity
            )
    else:
        return entity


@app.route('/data')
def data():
    return flask.jsonify(
        lazify_entity(rendered_trajectory, depth=2, lazy_keys=('children',))
    )


@app.route('/entity/<int:entity_id>')
def entity(entity_id):
    return flask.jsonify(entities[entity_id])


if __name__ == '__main__':
    trace = tracing.load(sys.argv[1])
    (rendered_trajectory, entities) = render_trajectory(trace.trajectory)
    app.run()
