#! /usr/bin/env python

import visdom
import argparse
import json
import os


# SAVING

def create_log_at(file_path, current_env, new_env=None):
    new_env = current_env if new_env is None else new_env
    vis = visdom.Visdom(env=current_env)

    data = json.loads(vis.get_window_data())
    if len(data) == 0:
        print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
        return

    file = open(file_path, 'w+')
    for datapoint in data.values():
        output = {
            'win': datapoint['id'],
            'eid': new_env,
            'opts': {}
        }

        if datapoint['type'] != "plot":
            output['data'] = [{'content': datapoint['content'], 'type': datapoint['type']}]
            if datapoint['height'] is not None:
                output['opts']['height'] = datapoint['height']
            if datapoint['width'] is not None:
                output['opts']['width'] = datapoint['width']
        else:
            output['data'] = datapoint['content']["data"]
            output['layout'] = datapoint['content']["layout"]

        to_write = json.dumps(["events", output])
        file.write(to_write + '\n')
    file.close()


def create_log(current_env, new_env=None):
    new_env = current_env if new_env is None else new_env
    dir_path = os.getcwd()
    if not os.path.exists(dir_path + '/log'):
        os.makedirs(dir_path + '/log')
    file_path = dir_path + '/log/' + new_env + '.log'
    create_log_at(file_path, current_env, new_env)


# LOADING

def load_log_at(path):
    visdom.Visdom().replay_log(path)


def load_log(env):
    dir_path = os.getcwd()
    load_log_at(dir_path + '/log/' + env + '.log')


def load_all_log():
    dir_path = os.getcwd() + '/log/'
    logs = os.listdir(dir_path)
    vis = visdom.Visdom()
    for log in logs:
        vis.replay_log(dir_path + log)


# MAIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Load and Save from Visdom'))
    parser.add_argument('-s', '--save', type=str, help='env_name', default='')
    parser.add_argument('-l', '--load', type=str, help='env_name', default='', nargs="?")
    parser.add_argument('-f', '--file', type=str, help='path_to_log_file', default='')
    args = parser.parse_args()

    if args.save is not '':
        if args.file is not '':
            create_log_at(args.file, args.save)
        else:
            create_log(args.save)

    if args.load is not '':
        if args.load == 'all':
            load_all_log()
        elif args.load is not None:
            load_log(args.load)
        elif args.file is not '':
            load_log_at(args.file)
