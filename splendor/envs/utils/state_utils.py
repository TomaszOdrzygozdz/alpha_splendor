from splendor.envs.mechanics.state_as_dict import StateAsDict

def clone(arg):
    if arg.type == 'state':
        return StateAsDict(arg).to_state()