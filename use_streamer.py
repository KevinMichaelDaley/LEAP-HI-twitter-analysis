from tweat import stream
from tweat import parse
import sys

if __name__ == '__main__':
    stream.stream_by_topic(parse.parse())
