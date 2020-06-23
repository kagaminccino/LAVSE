# https://github.com/theevann/visdom-save
# visdom-save

Basic python script to save and load locally visdom visualisations.

# Why

If you are also frustrated that you cannot load your saved viz easily,

If you have forgotten the option 'log_to_filename' and want to have a copy of you viz,

If you work on multiple machines and want to easily load all you viz on a new machine.

Check out this script !

# Use
By default save and load log files at `./log/env_name.log`.

To save environment `env_name` (optionally in a specific file `file_path`):
```
python vis.py --save env_name
python vis.py --save env_name --file file_path
```

To load environment `env_name`, load all environment in log directory, or load from file `file_path`:
```
python vis.py --load env_name
python vis.py --load all
python vis.py --load --file file_path
```

# Warning
Not saving `opts` for plots since we cannot get this info.
