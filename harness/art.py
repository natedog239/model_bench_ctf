"""ASCII art and animation assets for the TUI (Evil Wizard theme).

Everything here is plain ASCII (no non-cp1252 glyphs, so it renders on the
legacy Windows console). The TUI composes these into rich renderables.
"""

import random

# -----------------------------------------------------------------------------
# Landing: the evil wizard, skull face, arms raised, channeling power.
# -----------------------------------------------------------------------------
WIZARD_SUMMON = r"""
      \                                     /
       \         .-'''''-.        .-'''''-./
        \       /  ~   ~  \      /  power  \
         \     |  (O) (O)  |    | ~ rises ~ |
          \     \    ^    /      \  .---.  /
           '--.  \  '-'  /  .--.  \(     )/
               \  '-...-'  /    \  '-...-'
                \        _/      \_        /
                 '.    /   SKULL   \    .'
                   '-.| .-.  MAGE .-. |.-'
                      \|  '.(   ).'  |/
                       \    '-'-'    /
                        '.  =====  .'
                          '.  |  .'
                     ______|  |  |______
                    /      THE ARCHMAGE  \
                   /    OF   BENCHMARKS   \
                   '.__.-''''       ''''-.__.'
                        ||          ||
                       /||          ||\
                      /_||__________||_\
"""

# -----------------------------------------------------------------------------
# Running: compact wizard, facing RIGHT, one arm extended casting a spell.
# Rendered as lines; the extended-hand row is where the spell attaches.
# The '@' marks the hand -- the TUI replaces the region to its right with the
# animated spell that carries the current test name.
# -----------------------------------------------------------------------------
WIZARD_BATTLE = r"""
     .-~~~-.
    / _   _ \
   | (o) (o)|
   |   <>   |
    \  \_/  /
  .--'~~~~~'--.
 / THE ARCHMAGE\
|  .---------.  \
|  | ~ * ~ |  |  @======
 \ | * mage* |  |
  \'---------'  /
   |    |||    |
   |    |||    |
  /`    |||    `\
 /______|||_____\
    (___)   (___)
"""

# GAME OVER big block letters.
GAME_OVER = r"""
  ____    _    __  __ _____    _____     _______ ____
 / ___|  / \  |  \/  | ____|  / _ \ \   / / ____|  _ \
| |  _  / _ \ | |\/| |  _|   | | | \ \ / /|  _| | |_) |
| |_| |/ ___ \| |  | | |___  | |_| |\ V / | |___|  _ <
 \____/_/   \_\_|  |_|_____|  \___/  \_/  |_____|_| \_\
"""

# A little grinning skull for the results screen accents.
SKULL = r"""
   .-.
  (o o)
  | O |
   \_/
"""

# Glyphs for magical particle background.
_RUNE_GLYPHS = "*+.:x%#@=~^oO0"

# -----------------------------------------------------------------------------
# SPELLS
# Five named spells. Each spell wraps the CURRENTLY RUNNING TEST NAME so the
# spell literally carries the test toward the target. `left`/`right` are the
# decorations placed around the test text; `frames` are slow-cycling variants
# of the trailing beam so it shimmers without being frantic. `style` is a rich
# style for coloring.
# -----------------------------------------------------------------------------
SPELLS = {
    "fireball": {
        "left": "((*",
        "right": "*))~>",
        "frames": ["~~=>", "=~~>", "~=~>", "==~>"],
        "style": "bold red",
        "label": "F I R E B A L L",
    },
    "lightning": {
        "left": "-=[",
        "right": "]=-/\\>",
        "frames": ["/\\/>", "\\/\\>", "/z!>", "//\\>"],
        "style": "bold bright_yellow",
        "label": "L I G H T N I N G",
    },
    "frost": {
        "left": "*.:[",
        "right": "]:.*>>",
        "frames": ["**~>", "*.*>", ".**>", "*+*>"],
        "style": "bold bright_cyan",
        "label": "F R O S T   B O L T",
    },
    "arcane": {
        "left": "<@{",
        "right": "}@>~>",
        "frames": ["~o~>", "o~o>", "~O~>", "oOo>"],
        "style": "bold magenta",
        "label": "A R C A N E   M I S S I L E",
    },
    "void": {
        "left": "#%[",
        "right": "]%#=>",
        "frames": ["%#%>", "#%#>", "%%#>", "#o#>"],
        "style": "bold blue",
        "label": "V O I D   L A N C E",
    },
}

# Deterministic mapping from a test id to a spell, so the spell == the test.
_TEST_SPELL = {
    "test1_auth_gating": "arcane",
    "test2_string_manipulation": "fireball",
}
# Ordered fallback for any unmapped/extra tests.
_SPELL_ORDER = ["fireball", "lightning", "frost", "arcane", "void"]


def spell_for_test(test_id: str) -> dict:
    """Return the spell dict for a given test id (stable mapping)."""
    if test_id in _TEST_SPELL:
        key = _TEST_SPELL[test_id]
    else:
        # Hash the id into one of the five spells so new tests still get one.
        key = _SPELL_ORDER[hash(test_id) % len(_SPELL_ORDER)]
    return {**SPELLS[key], "key": key}


def render_spell(test_id: str, test_label: str, tick: int) -> str:
    """Build the spell string that carries the test name toward the target.

    Example: ((* auth_gating *))~>~~=>
    The trailing beam shimmers slowly via `frames`.
    """
    spell = spell_for_test(test_id)
    beam = spell["frames"][tick % len(spell["frames"])]
    return f"{spell['left']} {test_label} {spell['right']}{beam}"


def lines_of(block: str):
    return [ln for ln in block.split("\n") if ln.strip() != ""]


def raw_lines(block: str):
    """Keep internal blank lines but trim leading/trailing empties."""
    lines = block.split("\n")
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines


def wizard_summon_lines():
    return raw_lines(WIZARD_SUMMON)


def wizard_battle_lines():
    return raw_lines(WIZARD_BATTLE)


def game_over_lines():
    return raw_lines(GAME_OVER)


def skull_lines():
    return raw_lines(SKULL)


def rune_line(width: int, density: float = 0.08) -> str:
    chars = []
    for _ in range(width):
        chars.append(random.choice(_RUNE_GLYPHS) if random.random() < density else " ")
    return "".join(chars)


def rune_block(width: int, height: int, density: float = 0.08):
    return [rune_line(width, density) for _ in range(height)]
