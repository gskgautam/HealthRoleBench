import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile


@contextmanager
def private_text_artifact(path):
    """Publish a complete owner-only artifact without replacing an existing result."""
    if path.exists():
        raise FileExistsError(path)
    with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent) as temporary:
        yield temporary
        temporary.flush()
        os.link(temporary.name, path)
