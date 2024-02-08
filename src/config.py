import configparser
import argparse
import os


class App:
    file_path = None
    _conf = None

    @staticmethod
    def config():
        if App.file_path is None:
            print('Checking for alternative file path to .ini file')
            parser = argparse.ArgumentParser(
                description='Run bindNode24 for binding residue prediction')
            parser.add_argument('-f',
                                default="src/config.ini",
                                type=str,
                                help='File path to .ini file if it isn\'t config.ini',
                                metavar="file_path")
            args = parser.parse_args()
            print(f'Used filepath is: {App.file_path}')
        if App._conf is None:
            App._conf = configparser.ConfigParser()
            App._conf.read(App.file_path)

        return App._conf