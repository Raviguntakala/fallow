from service import build_user


class User:
    def __init__(self, name):
        self.name = name


def make_default():
    return build_user("anon")
