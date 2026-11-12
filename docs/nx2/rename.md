# Renaming inherited yuzu names

Lichii inherits code from yuzu through Eden and Eglantine. The yuzu project name is replaced
with Lichii in code, build files, paths, user-visible text and documentation. Copyright and
licence notices stay exactly as they are, and so does anything that existing user data,
caches or build directories depend on.

The rename is a codemod, `tools/rename/rename.py`, so it can be run again after an upstream
sync. Upstream Eglantine still uses the yuzu names.

## Running it

```sh
python3 tools/rename/rename.py apply           # every area
python3 tools/rename/rename.py apply --area build
python3 tools/rename/rename.py check -v        # what still says yuzu, by class
python3 tools/rename/rename.py check --strict  # fails on anything not classified
```

`apply` moves tracked paths with `git mv` and rewrites text line by line. Both are idempotent:
a second run changes nothing.

Each file in `tools/rename/areas/` is one area and holds three lists:

- `PATHS`: `(regex, replacement, options)` applied to tracked paths.
- `TEXT`: `(regex, replacement, options)` applied to each line. `options` may set `files`
  and `exclude` (fnmatch globs) and `in_urls` (rules skip URLs unless it is set).
- `KEEP`: `(regex, class, reason[, files])`. Text a KEEP pattern matches is never rewritten,
  whatever area the rule comes from, and `check` reports it under its class.

Never rewritten, by any rule: lines with `SPDX-FileCopyrightText` or a `.reuse/dep5`
copyright, licence files, and `externals/` (only the build option names in our own CMake glue
there are renamed).

## Classes

- **attribution**: copyright notices and statements of where the code comes from. Removing
  them would break the GPL.
- **compat**: on-disk names, file magics, protocol strings and old option names that existing
  installs depend on. Lichii keeps reading them.
- **pending**: anything else; `check --strict` fails while any remain.

## Build options

The CMake options are `LICHII_*` (`LICHII_TESTS`, `LICHII_CMD`, `LICHII_USE_BUNDLED_QT`, ...).
`CMakeModules/LegacyOptions.cmake` carries an old `-DYUZU_*` value, from the command line or
an existing cache, to the new name and removes the old entry, so existing build directories
and scripts keep their settings. The targets are `lichii` (output `lichii`), `lichii-cmd`
(output `eden-cli`), `lichii-room` and `lichii_room_standalone` (output `eden-room`).

## After an upstream sync

1. Merge upstream. Upstream files under `src/yuzu`, `src/yuzu_cmd` and the other old paths
   are renamed here, so git reports rename conflicts for files both sides changed: take
   upstream's change into the renamed file.
2. Run `python3 tools/rename/rename.py apply`, then `check --strict`.
3. Classify anything new that `check` lists: add a TEXT rule to the area that owns it, or a
   KEEP pattern when the name is attribution or compatibility data.
4. Build and commit the result as its own commit.

A branch that was opened before the rename can do the same: run `apply` on the branch, commit,
then merge nx2.
