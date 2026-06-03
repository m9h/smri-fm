"""Stub: only referenced by MultiModalUNetVAE.predict() (sliding-window
inference), which the asparagus bridge never calls — asparagus runs its own
eval loop. Raising keeps the unused path honest."""


def get_steps_for_sliding_window(*args, **kwargs):
    raise NotImplementedError(
        "yucca sliding-window inference is stubbed out in the asparagus bridge"
    )
