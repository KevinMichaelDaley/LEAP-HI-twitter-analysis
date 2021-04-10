from argparse import ArgumentParser
from getpass import getpass

def setup_parser(topic=True):

    parser = ArgumentParser(
        description="Tweat is a python3 library for treating tweets like meat.")

    parser.add_argument('host', type=str,
                        help="host for the mysql database")

    parser.add_argument('user',  type=str,
                        help="username for the mysql database")

    parser.add_argument('db', type=str,
                        help="name of the mysql database")
    if topic:
        parser.add_argument('topic', type=str,
                        help="a topic to search, e.g. '@vpostman'")
    
    passgroup = parser.add_mutually_exclusive_group(required=True)

    passgroup.add_argument('--nspass', type=str,
                           help="non-secure password for the mysql database")

    passgroup.add_argument('--spass', '--secure_pass',
                           action='store_true', dest='password',
                           help="secure password prompt for the mysql database")
    return parser


class ArgumentStore:
    # Each attribute of an ArgumentStore instance 
    # is a parsed argument.
    # (e.g. argstore.password contains the db password)
    pass


def parse(topic=True):
    
    parser = setup_parser(topic)

    argstore = ArgumentStore()
    parser.parse_args(namespace=argstore)

    if argstore.password:
        argstore.password = getpass()
        del argstore.nspass
        parsed_args = dict(vars(argstore))
        return parsed_args
    setattr(argstore, 'password', argstore.nspass)
    del argstore.nspass

    parsed_args = dict(vars(argstore))
    return parsed_args


if __name__ == '__main__':
    parsed = parse()
    print(', '.join("%s: %s" % item for item in parsed.items()))
    print(type(parsed))
    print(parsed['password'])
