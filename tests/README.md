# Tests Directory Guide
This is a brief guide/suggestion for how to organise your tests and hopefully never mix up your saved runs and which configs they correspond to!

## Setting Up new tests cases
1. Create a new directory inside tests named for your test case e.g. `tests/crazy_test/`.
2. Generate a `.ini` config inside your new test case folder.
3. Save corresponding `.npz` runs inside that folder.

The value in this method is that `.npz` are siblings with **only** their `.ini` config, and no others.

## Rules
1. Don't put save two `.ini` config files into 1 directory! 
2. Don't save `.npz` files to directories which don't also contain their `.ini`.