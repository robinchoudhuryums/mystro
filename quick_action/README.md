# Finder Quick Action: Master with MasterBench

Right-click one or more WAV/AIFF files in Finder and master them. Not yet tried on a Mac;
the steps below are the standard Automator route.

## Install

1. Open **Automator** and create a new **Quick Action**.
2. At the top set: *Workflow receives current* **audio files** *in* **Finder**.
3. Add the action **Run Shell Script**. Set *Shell* to `/bin/zsh` and *Pass input* to
   **as arguments**.
4. Replace the script with this one line, using the path to your checkout:

   ```bash
   MASTERBENCH_REPO="$HOME/Developer/mystro" "$HOME/Developer/mystro/quick_action/master-with-masterbench.sh" "$@"
   ```

5. Save it as **Master with MasterBench**. It appears under *Quick Actions* in Finder's
   right-click menu.

## What it does

- One file: `masterbench master FILE --notify`, using the default genre's default reference.
- Several files: `masterbench batch FILES --notify`, which also writes the consistency report.

The genre comes from `default_genre` in `settings.toml`. For a different genre or reference,
use the command line or the web UI.

Clicking the notification opens the run in the UI when `terminal-notifier` is installed
(`brew install terminal-notifier`) and `masterbench ui` is running; without it you get a
plain notification.
