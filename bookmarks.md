## Bookmarks and Stuff

Here are things that I didn't invent, but I find myself constantly referring back to.

---

### The Catherine Project

Very rarely does something come along that makes me think about lifelong learning, or rather the act of _learning_ itself. [The Catherine Project](https://catherineproject.org/) is definitely one such initiative. How can critical thinking be taught or encouraged? How should we introspectively write about ourselves and the thoughts of others?

More importantly, below is a reproduction of the project's writing guidelines, something which I think is actually more important than the reading lists itself (and for some reason is somewhat hard to find):

* We write in order to enrich and focus our thinking. A piece of writing is not a performance intended to impress others or be used for evaluation. We seek to think seriously rather than to communicate "correct" thoughts or to devise phrases that sound scholarly or artful.
* Writing is a work of the mind. In order to write well about a question or idea, one must first spend some time thinking. Reread, think and take notes. Use writing as a way to think through your questions or ideas, and discover new ones.
* In deciding what to write about, no question is too simple or elemental.
* Use the text. You might attempt to formulate a question that the text raises for you, or to respond to such a question. You may object to or be puzzled by the way the text presents a topic. You may muse on something that the text brough to mind. In other words, you may write anything you think might help you - and your reader - think about the text.
* Write about something you care about. Try to find something that grips you, whether this is the text itself, a particular question or problem raised in the text, or something you love or hate in the text.
* Meaningful writing takes time. The more ripe it is, the better it is. Give yourself enough time to write something, set it aside, and return to it. Don't be anxious if your ideas change in time: this means you learned something through writing.
* That said, don't tarry. The virtue of a short assignment is that it be _done_. It may be useful to start writing even ifyou don't know what you're going to say. A little insight or a short and simple reflection is worth writing. In fact writing it down may help it grow into something grander in its own time.
* When at a loss, start writing about something else, and eventually inspiration may lead you back to where you lost the thread. Or start writing about why you're having trouble thinking of what to say. Or set the writing aside for a while and then come back. Don't stare at the page racking your brain for the perfect next sentence.
* Remain flexible. We don't approach serious conversation about a text as a battle or a debate. Avoid writing about the text from a predetermined position. Consider any side of the question that occurs to you, with openness and candor.
* Be generous with your reader. Try first to say what you mean. Then consider who your reader is and how they will understand you. This can take practice, as we often think our readers have a much more sophisticated or complex understanding than they do in fact.
* Speak plainly. Whenever you cite the text, offer page or line numbers or any other shared access point, as this allows others to find your place in the text.

### A 'local' `.gitignore`

Sometimes you want to ignore files and not push them to VSC but also don't want `git` to even possibly consider it. To do this, you can update `.git/info/exclude`

### `uv` Script Mode

See [here](https://docs.astral.sh/uv/guides/scripts/#creating-a-python-script)

`uv` now supports script mode. This is quite amazing for one-off scripts rather than publishing and overloading pypi, you can just share a single script and `uv` will correctly pull depdencies (reminds me of [flit](https://flit.pypa.io/en/stable/)). 

**Usage**

Start by using `--script` mode

```sh
uv init --script example.py --python 3.12
```

Then you can add dependencies. This will update the `example.py` with the appropriate header

```sh
uv add --script example.py 'requests<3' 'rich'
```

If one then follows up by adding the shebang

```py
#!/usr/bin/env -S uv run --script
```

So that the resulting file looks something like

```py
#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint
...
```

Then the script will be completely runnable as a normal `cli` command!

Unfortunately using it interactively isn't so straightforward. Though a recipe like:

```sh
$ "$(uv python find --script script.py)"
>>> from script import X
```

Should work

### Architecture Decision Record

My go-to template is the [Alexandrian Pattern]([https://github.com/jamesmh/architecture_decision_record/blob/master/adr_template_for_alexandrian_pattern.md](https://github.com/joelparkerhenderson/architecture-decision-record/tree/main/locales/en/templates/decision-record-template-for-alexandrian-pattern)), which is based on [Design Pattern from Christopher Alexander](https://en.wikipedia.org/wiki/Design_pattern)

#### Introduction

* Prologue (Summary)
* Discussion (Context)
* Solution (Decision)
* Consequences (Results)

#### Specifics

* Prologue (Summary)
  * Statement to summarize:
    * In the context of (use case)<br>
      facing (concern)<br>
      we decided for (option)<br>
      to achieve (quality)<br>
      accepting (downside).
* Discussion (Context)
  * Explains the forces at play (technical, political, social, project).
  * This is the story explaining the problem we are looking to resolve.
* Solution
  * Explains how the decision will solve the problem.
* Consequences
  * Explains the results of the decision over the long term.
  * Did it work, not work, was changed, upgraded, etc.
 
---

### HTML Template

[Pico](https://picocss.com/) has proven to me to be a good enough HTML template with sensible defaults. It can be classless if you want it to be, or you can have some extras on top. 

It provides you with a good starter template.

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light dark">
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"
    >
    <title>Hello world!</title>
  </head>
  <body>
    <main class="container">
      <h1>Hello world!</h1>
    </main>
  </body>
</html>
```

Color scheme overrides can be done just by adding an attribute to the html tag, e.g. `<html data-theme="light">`.

---

### Organising Stuff

I keep coming back to [Johnny.Decimal](https://johnnydecimal.com/). It has been great for personal, admin things - though I haven't used it for work or programming related projects. On that front still seeing if I can do better. This blog site "setup" (or lack thereof) has definitely been influenced by Johnny Decimal.
