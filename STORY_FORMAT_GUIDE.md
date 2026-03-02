# Story text format guide

Use this when cleaning or checking story `.txt` files (e.g. from PDF extraction).

---

## 1. Contractions (no space around apostrophe)

**Rule:** Contractions are one word. No space before or after the apostrophe.

| Wrong        | Right   |
|-------------|---------|
| I 'm        | I'm     |
| it 's       | it's    |
| It 's       | It's    |
| I 've       | I've    |
| I 'll       | I'll    |
| you 're     | you're  |
| You 're     | You're  |
| we 'll      | we'll   |
| We 'll      | We'll   |
| don 't      | don't   |
| what 's      | what's  |
| What 's     | What's  |
| that 's     | that's  |
| That 's     | That's  |
| she 's      | she's   |
| he 's       | he's    |
| they 're    | they're |
| we 've      | we've   |
| could n't   | couldn't |
| would n't   | wouldn't |

**Check for:** Any `letter ' letter` pattern in the middle of a word.

---

## 2. Dialogue quotes (opening)

**Rule:** No space after the opening single quote.

| Wrong      | Right    |
|------------|----------|
| ' It's     | 'It's    |
| ' I        | 'I       |
| ' Oh       | 'Oh      |
| ' What     | 'What    |

**Pattern:** Opening quote is always `'Word` (e.g. `'It's`, `'Oh`, `'What`).

---

## 3. Dialogue quotes (closing)

**Rule:** No space between the last character of the dialogue and the closing quote. One space after the closing quote before the next word.

| Wrong           | Right          |
|----------------|----------------|
| word, ' she    | word,' she     |
| word? ' he     | word?' he      |
| word! ' she    | word!' she     |
| again ' ; and  | again'; and    |

**Patterns:**
- After comma: `,' she` not `, 'she`
- After ? or !: `?' he` not `? 'he`; `!' she` not `! 'she`
- Before semicolon: `'; and` not ` '; and`

---

## 4. PDF hyphenation (rejoin split words)

**Rule:** Words split across lines in the PDF often end up as `word- next` or `word- next`. Rejoin into one word (or a proper compound) when it’s a single concept.

| Wrong          | Right         |
|----------------|---------------|
| hence- forth   | henceforth    |
| how- ever      | however       |
| under- stand   | understand    |
| examina- tion  | examination   |
| Every- thing   | Everything    |
| New- foundland | Newfoundland  |
| con- fused     | confused      |
| dis- quieting  | disquieting   |
| Un- fortunately| Unfortunately |
| hiding- place  | hiding-place  |

**Keep as hyphenated** when it’s a normal compound:  
e.g. `night-lights`, `heart-gripping`, `spring-cleaning`, `drawing-room`, `matter-of-fact`.

---

## 5. Dirty PDFs: weird symbols and noise

**Rule:** PDFs from scans, old books, or bad encoding often produce **weird symbols**, replacement characters, and stray punctuation. Remove or replace these in a first pass so the rest of the guide applies to clean text.

### 5a. Common weird symbols (remove or replace)

| Symbol / pattern | Meaning / cause | Action |
|------------------|-----------------|--------|
| `�` (replacement char) | Invalid or unknown Unicode | Remove, or replace with correct letter if obvious |
| `€` in the middle of a word | Wrong encoding (e.g. `e` misread) | Replace with correct letter (e.g. `st€p` → `step`) |
| `�` (diamond question mark) | Same as above | Remove or fix |
| Stray `\|` or `¦` in prose | Pipe/break misread or artifact | Remove |
| `*` or `•` in the middle of a sentence | Bullet/list artifact | Remove if not part of the story |
| `#` in the middle of a word | Number sign misread | Replace with correct character (e.g. `s#e` → `she`) |
| `@` in prose | At-sign misread (often `a`) | Replace with `a` if it's clearly wrong |
| `0` (zero) in place of `O` (letter) | OCR confusion | Replace when it's clearly the letter O |
| `1` (one) in place of `l` or `I` | Same | Replace when clearly letter |
| Multiple spaces or `  ` | Extra spaces from layout | Collapse to single space |

### 5b. Encoding and mojibake

- **Mojibake:** If you see sequences like `Ã©` instead of `é`, or `â€™` instead of `'`, the text was decoded with the wrong encoding (e.g. UTF-8 read as Latin-1). Re-open or re-extract the PDF with the correct encoding (usually UTF-8), or run a mojibake fix (e.g. `ftfy` in Python) before cleaning.
- **Mixed encodings:** Some PDFs have one encoding in one place and another elsewhere. Fix the most common pattern first, then do a pass for remaining weird sequences.

### 5c. First pass for dirty PDF output

1. **Strip or replace** all replacement characters (e.g. � U+FFFD).
2. **Fix** common symbol-for-letter swaps (e.g. `€`→`e`, `#`→`h`, `@`→`a`) where the intent is clear.
3. **Remove** stray bullets, pipes, and control characters that are not part of the story.
4. **Normalize spaces:** collapse multiple spaces/tabs to a single space; trim leading/trailing spaces (see §10).
5. **Then** apply the rest of this guide (contractions, quotes, hyphenation, OCR word fixes, etc.).

**Check for:** Any character that is not a normal letter, digit, or standard punctuation (e.g. `'` `"` `,` `.` `?` `!` `;` `:` `-` `—`). If it looks like noise, remove or replace it.

---

## 6. OCR / scanning errors

**Rule:** Fix obvious misreads from PDF/OCR (after cleaning weird symbols per §5).

| Wrong   | Right   |
|---------|---------|
| ofifing | offing  |
| corrie  | come    |
| jienying| denying|
| ereat   | great  |
| st€p    | step   |

---

## 7. Titles (Mr., Mrs., Ms., Dr.)

**Rule:** Abbreviations for titles always have a **space** after the period, before the name.

| Wrong       | Right        |
|-------------|--------------|
| Mrs.Darling | Mrs. Darling |
| Mr.Darling  | Mr. Darling  |
| Dr.Smith    | Dr. Smith    |
| Ms.Jones    | Ms. Jones    |

**Pattern:** `Title. Name` (e.g. `Mrs. Darling`). No space is a common PDF/OCR error.

---

## 8. Punctuation (no space before ? ! . , ; :)

**Rule:** Punctuation marks are attached to the last word—**no space** before `?` `!` `.` `,` `;` `:`.

| Wrong     | Right    |
|-----------|----------|
| crying ?  | crying?  |
| right !   | right!   |
| yes .     | yes.     |
| hello ,   | hello,   |
| word ;    | word;    |

**Examples:** `'why are you crying?'` not `'why are you crying ?'`; `'How awful!'` not `'How awful !'`. A space before `?` or `!` often comes from old typesetting or PDF extraction—remove it.

---

## 9. Dialogue split across lines (one speech = one unit)

**Rule:** In the novel, one person’s speech is often a single sentence or several sentences. If that speech is split across two or more lines (e.g. by PDF line breaks), treat it as **one unit**, not two.

**Example — wrong (split into two):**
```
First:  'Yes, but at what a cost I 
Second: By depriving the children of ten minutes of delight.'
```

**Correct:** Keep the whole speech together as one line/one segment:
```
'Yes, but at what a cost! By depriving the children of ten minutes of delight.'
```

**What to do:**
- **In the .txt file:** Rejoin lines so that each line is one complete speech (from opening `'` to closing `'`). If a speech was broken across lines, merge those lines into one.
- **When splitting into “sentences” or chunks** (e.g. for a vector store or search): Never split in the middle of dialogue. Split only:
  - after a **closing quote** `'` (end of speech), or
  - at **narration** (text that is not inside quotes).
- **One speech = one segment.** Everything from `'` to `'` is one unit, even if it contains multiple sentences (e.g. `'Yes, but at what a cost! By depriving the children of ten minutes of delight.'`).

**Quick check:** If a line ends with `'` and the next line starts with a capital letter but no `'`, the previous line is probably an incomplete speech—merge it with the next line(s) until you see the closing `'`.

---

## 10. One paragraph = one line (trailing spaces and alignment)

**Rule:** Each **paragraph** in the story is one **line** in the file. Put all the sentences that belong to the same paragraph on that single line (with normal spaces between sentences). Do not put one sentence per line unless each sentence is its own paragraph in the book.

- **Yes:** All sentences from one paragraph → one line.  
  Example: a paragraph of three sentences stays as one line:  
  `She was not alarmed to see a stranger. She was only interested. She sat up in bed.`
- **No:** One sentence per line (unless the book has that sentence as its own paragraph).  
  Do not break a paragraph into multiple lines like:
  ```
  She was not alarmed to see a stranger.
  She was only interested.
  She sat up in bed.
  ```
  when those three sentences form one paragraph in the novel.

**How to tell which sentences form one paragraph (and how to tell an editor/AI):**

- **From the original book or PDF:** A new paragraph in the source usually starts on a new line, often with indentation or a blank line above it. So: one block of text without a paragraph break in the source = one paragraph = one line in your .txt.
- **When asking someone to fix or merge lines,** you can say clearly which sentences belong together, for example:
  - *“Sentences 1–3 are one paragraph; merge them into one line.”*
  - *“From ‘She was not alarmed’ through ‘She sat up in bed.’ is one paragraph.”*
  - Or paste the snippet and say: *“These three sentences are one paragraph—please make them one line.”*
- **If you’re not sure:** When in doubt, treat a short block of related sentences (same speaker or same moment) as one paragraph; you can always split later if you check the printed book and see a paragraph break.

Also: no trailing spaces at the end of lines; keep line breaks consistent (one line per paragraph).

---

## 11. Quick checklist

When checking a story file:

- [ ] **No weird symbols:** No �, € in words, stray | • # @ (unless part of story); run first pass (§5) on dirty PDF output
- [ ] No space in contractions: `I'm` not `I 'm`, `it's` not `it 's`, etc.
- [ ] Opening quote: `'Word` not `' Word`
- [ ] Closing quote: `word,' she` and `word?' he` / `word!' she` (no space before `'`)
- [ ] No PDF hyphenation: rejoin `word- next` when it’s one word
- [ ] No OCR typos (e.g. ofifing → offing, jienying → denying); see §6
- [ ] Titles: `Mrs. Darling` not `Mrs.Darling` (space after the period)
- [ ] No space before punctuation: `crying?` not `crying ?`, `right!` not `right !`
- [ ] No trailing spaces at line ends
- [ ] Consistent punctuation (e.g. no stray `'` at end of line)
- [ ] Dialogue not split in the middle: one speech = one line/one chunk when splitting

---

## Why this format works well for vector stores

Yes—keeping your story files in this format is a good approach for vector stores (embeddings, RAG, semantic search).

- **One paragraph = one line** → When you ingest, you can use **one line = one chunk**. Each chunk is a natural, coherent unit (a paragraph or a full speech), so embeddings capture meaning better than random splits.
- **No mid-dialogue splits** → Chunks don’t cut in the middle of a speech, so retrieval returns complete quotes instead of broken fragments.
- **Clean text** → Consistent punctuation, no weird symbols or dirty-PDF noise (§5), no OCR typos (§6), and no broken words improve tokenization and embedding quality. The model sees normal prose, not “I 'm” or “crying ?”.
- **Stable chunking** → Same file always produces the same chunks; no surprise boundaries and easier to debug and cite.

When ingesting: read the file line by line and treat each line as one chunk (or merge a fixed number of lines per chunk if you need larger chunks). Avoid splitting a line in the middle.

---

*Apply this guide to all story `.txt` files for consistent, clean text (e.g. for vector store or search).*
