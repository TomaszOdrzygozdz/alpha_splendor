"""Messages used in the coroutine API for Agent <-> Network communication.

Message types play the role of type tags, so we can distinguish which is which
in a stream of messages using isinstance().
"""


import collections


PredictRequest = collections.namedtuple('PredictRequest', ['inputs'])
Action = collections.namedtuple('Action', ['action'])
Episode = collections.namedtuple(
    'Episode', ['episode']
)
