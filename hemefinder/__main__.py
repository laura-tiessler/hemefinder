from .cli import client
from .hemefinder import hemefinder


def main():
    """
    Execute the program.
    """
    args = client()
    hemefinder(**args)


if __name__ == '__main__':
    main()
