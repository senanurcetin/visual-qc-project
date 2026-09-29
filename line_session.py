"""The visitor's line control state, kept in the signed session cookie."""
import time

from flask import session

import line_sim


def line_state():
    state = session.get("line")
    if not line_sim.is_valid(state):
        state = line_sim.new_state(time.time())
        session["line"] = state
        session.permanent = True
    return state
