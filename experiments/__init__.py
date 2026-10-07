import sys

# Windows consoles often use a legacy code page (cp1252), which cannot print the Greek letters in
# algorithm names ("MC constant-α"). Force UTF-8 so every script works the same on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a regular text stream (e.g. captured by pytest)
        pass
