"""Ensure Claude Code settings have Opus model set."""
import json, os, sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/.claude/settings.json')
s = json.load(open(path)) if os.path.exists(path) else {}
s['model'] = 'claude-opus-4-6'
json.dump(s, open(path, 'w'))
print(f'Opus model set in {path}')
