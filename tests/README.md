# Tests Directory Guide
This is a brief guide/suggestion for how to organise your tests and hopefully never mix up your save files.

## Setting Up new tests cases
1. Create a new directory inside tests named for your test case e.g. `tests/basic_tests/`.
2. Initialise a [Saver class](src/SaverLoader.py) with that directory, thus automatically creating subdirs: runs, predictions, readouts, and ccs.
2. Generate a `.ini` config inside your new test case directory e.g. `tests/basic_tests/basic_test_case.ini`.
3. Use your [Saver class](src/SaverLoader.py) to save runs, predictions, readouts, and ccs, for which the saver class will automatically organise them into their corresponding directories.

Since the [load_npzs function](src/SaverLoader.py) does distinguish between `.npz` save file data, organising in this way makes its use easier.
Moreover, this ensures that all the saved files correspond with the `.ini` config file in their parent directory.

## Rules
1. Don't save two `.ini` config files into 1 directory! 
2. Don't save `.npz` files to directories/subdirectories which don't also contain their `.ini`.
3. Once you've generated data from a given `.ini` config, treat it as though it's read-only (don't modify it to maintain a paper-trail).

# Example
Below shows an example file structure:
```
tests
├── basic_tests
│   ├── basic_test_case.ini
│   ├── ccs
│   │   └── ...
│   ├── predictions
│   │   └── ...
│   ├── readouts
│   │   └── ...
│   └── runs
│     └── basic_run.npz
└── other_tests
    ├── other_test_config.ini
    ├── ....
    └── runs
      └── other_run.npz
```
