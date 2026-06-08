import requests
from typing import Optional


def run(value: Optional[int]):
    return requests.post(str(value))
