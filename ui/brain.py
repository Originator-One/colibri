#!/usr/bin/env python3
"""The chat's second brain (a fork addition): search and read the Markdown notes in one
folder, such as an Obsidian vault, for the page's note tools (note_search, note_read).

    /brain/status                      {"on", "vault", "notes"}
    /brain/search?q=&n=                the best notes, a line or two from each
    /brain/read?note=&find=&max=       one note, whole or only the lines about `find`

Read only, and only inside that folder: a note is opened through the index this module
builds by walking the folder, never from a path the page sends, so nothing outside it can
be named. It skips the app's folders (.obsidian, .trash) and the raw material (raw-sources,
sources), so answers come from the notes themselves, not from the files they were made from.

Every character the model reads costs prompt time (on an engine that streams its experts
from the SSD, about a second for every five or six tokens), so the text is cut short and
tidied: Obsidian links become their text, source tags like [gmail, Mar 2026] and bold marks
go, and front matter is dropped. The folder comes from --brain or COLIBRI_BRAIN (server.py);
with neither, /brain/status says off and the page hides its note tools. Standard library only."""
import math, os, re, threading, time

ROOT = ""            # the notes folder, a real path; "" = off
SKIP = {".obsidian", ".trash", ".git", "node_modules", "raw-sources", "sources", "copilot", "templates", "_tools"}
RESCAN_S = 15        # a search or read looks for new, changed and deleted notes at most this often
SNIP = 170           # characters per line a search shows
MAX_READ = 1800      # characters a read returns by default

_lock = threading.Lock()
_notes = {}          # relative path -> note
_scanned = 0.0

STOP = set("""a about after all also am an and any are as at be been before being but by can could
did do does doing for from get got had has have how i if in into is it its just know let me more
most my no not now of on or our out over please say said should show so some tell than that the
their them then there these they this those to up us was we were what whats when where which while
who whom why will with would you your note notes""".split())

_FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)
_WIKI = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
_MDLINK = re.compile(r"\[([^\]]+)\]\((?:[^)\s]+)\)")
_TAG = re.compile(r"\s*\[(?:file:|web:|gmail|chatgpt|me\b|work|rillc|budget|site|presify-ws|personal)[^\]\n]*\]")
_BOLD = re.compile(r"\*\*|__")
_WORD = re.compile(r"[\w][\w'.-]*\w|\w", re.U)


def _clean(text):
    text = _FRONT.sub("", text.replace("\r\n", "\n"))
    text = _WIKI.sub(lambda m: m.group(2) or os.path.basename(m.group(1)), text)
    text = _MDLINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    text = _BOLD.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _aka(raw):
    """the names a note also goes by, from `aka:` or `aliases:` in its front matter"""
    m = _FRONT.match(raw.replace("\r\n", "\n"))
    if not m:
        return []
    out = []
    for line in m.group(0).splitlines():
        k, _, v = line.partition(":")
        if k.strip().lower() in ("aka", "aliases", "alias"):
            out += [x.strip().strip("'\"") for x in v.strip().strip("[]").split(",") if x.strip()]
    return out


def _scan():
    """the index, refreshed when it is older than RESCAN_S"""
    global _scanned
    if not ROOT:
        return {}
    with _lock:
        if time.time() - _scanned < RESCAN_S:
            return _notes
        seen = set()
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in SKIP and not x.startswith(".")]
            for f in files:
                if not f.endswith(".md") or f.startswith("."):
                    continue
                p = os.path.join(d, f)
                real = os.path.realpath(p)
                if not real.startswith(ROOT + os.sep):   # a link that leads out of the folder
                    continue
                rel = os.path.relpath(p, ROOT)
                seen.add(rel)
                try:
                    mt = os.path.getmtime(p)
                    if rel in _notes and _notes[rel]["mtime"] == mt:
                        continue
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        raw = fh.read()
                except OSError:
                    continue
                text = _clean(raw)
                _notes[rel] = {"rel": rel, "title": f[:-3], "aka": _aka(raw), "mtime": mt,
                               "text": text, "low": text.lower()}
        for rel in list(_notes):
            if rel not in seen:
                del _notes[rel]
        _scanned = time.time()
        return _notes


def _terms(q):
    out = []
    for w in _WORD.findall((q or "").lower()):
        w = w.strip(".'-")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                       # birthdays finds birthday, and birthday finds both
        if (len(w) >= 2 or w.isdigit()) and w not in STOP and w not in out:
            out.append(w)
    return out[:12]


def _idf(notes, terms):
    n = len(notes) or 1
    return {t: math.log((n + 1) / (sum(1 for x in notes.values() if t in x["low"]) + 0.5)) for t in terms}


def _score(note, terms, idf):
    s, hit = 0.0, 0
    names = [note["title"].lower()] + [a.lower() for a in note["aka"]]
    for t in terms:
        tf = note["low"].count(t)
        named = any(t in nm for nm in names)
        if tf or named:
            hit += 1
            s += idf[t] * (1 + math.log(1 + tf)) + (3 * idf[t] if named else 0)
    if not hit:
        return 0.0
    s *= (hit / len(terms)) ** 1.5           # a note with every word beats one that repeats one word
    if note["rel"] in ("log.md", "index.md") or note["rel"].startswith("wiki/health-checks/"):
        s *= 0.4                              # the change log, the catalog and the audits: the pages themselves first
    return s


def _lines(note, terms, idf):
    """the note's lines, each with its score and the heading it sits under. A word the note's
    name already holds counts little in a line: in Birthdays, the line about Audrey is the one
    that answers "Audrey's birthday", not the one that says "birthday" again."""
    names = [note["title"].lower()] + [a.lower() for a in note["aka"]]
    w = {t: idf[t] * (0.3 if any(t in nm for nm in names) else 1.0) for t in terms}
    out, head = [], ""
    for i, line in enumerate(note["text"].splitlines()):
        st = line.strip()
        if not st:
            continue
        if st.startswith("#"):
            head = st.lstrip("#").strip()
            continue
        low = st.lower()
        sc = sum(w[t] for t in terms if t in low)
        out.append((sc, i, head, st))
    return out


def _cut(line, terms, width):
    if len(line) <= width:
        return line
    low = line.lower()
    at = min([low.find(t) for t in terms if t in low] or [0])
    start = max(0, min(at - width // 3, len(line) - width))
    return ("..." if start else "") + line[start:start + width].strip() + ("..." if start + width < len(line) else "")


def status():
    notes = _scan()
    return {"on": bool(ROOT), "vault": os.path.basename(ROOT) if ROOT else "", "notes": len(notes)}


def search(q, n=4):
    notes = _scan()
    if not ROOT:
        raise ValueError("no notes folder (start the UI with --brain FOLDER)")
    terms = _terms(q)
    if not terms:
        raise ValueError("empty search")
    n = max(1, min(int(n), 8))
    idf = _idf(notes, terms)
    ranked = sorted(((_score(x, terms, idf), x) for x in notes.values()), key=lambda p: -p[0])
    results = []
    for sc, x in ranked[:n]:
        if sc <= 0:
            break
        best = sorted((l for l in _lines(x, terms, idf) if l[0] > 0), key=lambda l: (-l[0], l[1]))[:2]
        best.sort(key=lambda l: l[1])
        results.append({"note": x["title"], "path": x["rel"], "lines": [_cut(l[3], terms, SNIP) for l in best]})
    return {"query": q, "results": results}


def _find_note(notes, name):
    name = (name or "").strip().strip("[]").strip()
    if name.lower().endswith(".md"):
        name = name[:-3]
    low = name.lower()
    if not low:
        return None, False
    for x in notes.values():
        if x["rel"][:-3].lower() == low or x["title"].lower() == low:
            return x, False
    for x in notes.values():
        if any(a.lower() == low for a in x["aka"]):
            return x, False
    terms = _terms(name)
    if not terms:
        return None, False
    idf = _idf(notes, terms)
    sc, x = max(((_score(x, terms, idf), x) for x in notes.values()), key=lambda p: p[0], default=(0, None))
    return (x, True) if sc > 0 else (None, False)


def read(note, find="", max_chars=MAX_READ):
    notes = _scan()
    if not ROOT:
        raise ValueError("no notes folder (start the UI with --brain FOLDER)")
    x, guessed = _find_note(notes, note)
    if x is None:
        raise ValueError('no note called "%s": search for it first' % note)
    max_chars = max(300, min(int(max_chars), 6000))
    terms = _terms(find)
    if terms:
        idf = _idf(notes, terms)
        keep, last = [], None
        for sc, i, head, line in _lines(x, terms, idf):
            if sc <= 0:
                continue
            if head and head != last:
                keep.append("## " + head)
                last = head
            keep.append(line)
        body = "\n".join(keep) if keep else "(nothing in this note mentions %s)" % find
    else:
        body = x["text"]
    cut = len(body) > max_chars
    if cut:
        body = body[:max_chars].rsplit("\n", 1)[0] + "\n[cut: %d more characters. Read it again with find set to the part you need.]" % (len(body) - max_chars)
    return {"note": x["title"], "path": x["rel"], "guessed": guessed, "text": body, "truncated": cut}
