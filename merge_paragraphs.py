"""Merge story file: split by --- lines, merge each block to one line (Statement 9)."""
import sys

def is_marker(line):
    s = line.strip()
    if not s or len(s) < 3:
        return False
    # Treat any line of 3+ dashes (ASCII or Unicode) as paragraph separator
    return all(c in '- \t\u2013\u2014' for c in s)

def merge_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n').rstrip('\r') for line in f]

    blocks = []
    current = []
    for line in lines:
        if is_marker(line):
            if current:
                merged = ' '.join(ln.strip() for ln in current if ln.strip())
                if merged:
                    blocks.append(merged)
            current = []
        else:
            current.append(line)
    if current:
        merged = ' '.join(ln.strip() for ln in current if ln.strip())
        if merged:
            blocks.append(merged)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(blocks))
    print(f'Merged {len(blocks)} paragraphs in {path}')

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else r'Stories/COME AWAY, COME AWAY!.txt'
    merge_file(path)
