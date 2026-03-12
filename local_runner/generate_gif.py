#!/usr/bin/env python3
"""
Generate a retro terminal GIF animation from curated Zork gameplay highlights.
Produces a green-on-black CRT-style animation suitable for GitHub READMEs.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap

# --- Terminal dimensions ---
TERM_W = 620
TERM_H = 420
PADDING_X = 16
PADDING_Y = 12
TITLE_BAR_H = 28
SCREEN_Y = TITLE_BAR_H + 4
SCREEN_H = TERM_H - SCREEN_Y - 6
LINE_H = 16
FONT_SIZE = 13
CHARS_PER_LINE = 64

# --- Colors ---
BG = (10, 10, 10)         # Screen background
FRAME_BG = (26, 26, 26)   # Window frame
TITLE_BG = (42, 42, 42)   # Title bar
GREEN = (51, 255, 51)     # Bright green — game text
DIM_GREEN = (68, 119, 68) # Dim green — transitions
THOUGHT_GREEN = (100, 180, 100)  # Agent reasoning
ACTION_CYAN = (0, 255, 170)  # Agent actions
YELLOW = (255, 204, 0)    # Valid actions hints
GOLD = (255, 170, 0)      # Score/reward
RED = (255, 102, 102)     # Errors
WHITE = (200, 200, 200)   # Title text
GREY = (136, 136, 136)    # Title text dim

COLORS = {
    "game": GREEN,
    "thought": THOUGHT_GREEN,
    "action": ACTION_CYAN,
    "header": GREEN,
    "score": YELLOW,
    "reward": GOLD,
    "error": RED,
    "dim": DIM_GREEN,
    "blank": GREEN,
}


def load_font(size):
    """Load a monospace font."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_font_bold(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return load_font(size)


def draw_rounded_rect(draw, xy, radius, fill):
    x0, y0, x1, y1 = xy
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.pieslice([x0, y0, x0 + 2*radius, y0 + 2*radius], 180, 270, fill=fill)
    draw.pieslice([x1 - 2*radius, y0, x1, y0 + 2*radius], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2*radius, x0 + 2*radius, y1], 90, 180, fill=fill)
    draw.pieslice([x1 - 2*radius, y1 - 2*radius, x1, y1], 0, 90, fill=fill)


def make_base_frame(font, title_font):
    """Create the terminal frame (reusable base)."""
    img = Image.new("RGB", (TERM_W, TERM_H), FRAME_BG)
    draw = ImageDraw.Draw(img)

    # Rounded frame
    draw_rounded_rect(draw, (0, 0, TERM_W, TERM_H), 8, FRAME_BG)

    # Title bar
    draw_rounded_rect(draw, (0, 0, TERM_W, TITLE_BAR_H), 8, TITLE_BG)
    draw.rectangle([0, 14, TERM_W, TITLE_BAR_H], fill=TITLE_BG)

    # Traffic lights
    draw.ellipse([12, 8, 24, 20], fill=(255, 95, 87))
    draw.ellipse([30, 8, 42, 20], fill=(254, 188, 46))
    draw.ellipse([48, 8, 60, 20], fill=(40, 200, 64))

    # Title text
    title = "agentic-zork — Qwen 3.5 4B playing Zork I"
    bbox = title_font.getbbox(title)
    tw = bbox[2] - bbox[0]
    draw.text(((TERM_W - tw) // 2, 7), title, fill=GREY, font=title_font)

    # Screen area
    draw.rectangle([4, SCREEN_Y, TERM_W - 4, TERM_H - 4], fill=BG)

    return img


def render_frame(base_img, lines_to_show, font, show_cursor=True):
    """Render a frame with the given visible lines."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    # Clear screen area
    draw.rectangle([5, SCREEN_Y + 1, TERM_W - 5, TERM_H - 5], fill=BG)

    # Calculate visible area
    max_visible = (SCREEN_H - 2 * PADDING_Y) // LINE_H

    # Show last N lines (scrolling effect)
    visible = lines_to_show[-max_visible:] if len(lines_to_show) > max_visible else lines_to_show

    for i, (ltype, text) in enumerate(visible):
        color = COLORS.get(ltype, GREEN)
        y = SCREEN_Y + PADDING_Y + i * LINE_H
        if y + LINE_H > TERM_H - 5:
            break
        draw.text((PADDING_X + 4, y), text, fill=color, font=font)

    # Blinking cursor on last line
    if show_cursor and visible:
        last_line = visible[-1][1]
        cursor_x = PADDING_X + 4 + len(last_line) * 7.8
        cursor_y = SCREEN_Y + PADDING_Y + (len(visible) - 1) * LINE_H
        if cursor_y + LINE_H < TERM_H - 5:
            draw.rectangle(
                [cursor_x, cursor_y + 1, cursor_x + 8, cursor_y + LINE_H - 2],
                fill=GREEN
            )

    # Subtle scanline effect
    for y in range(SCREEN_Y, TERM_H - 4, 3):
        draw.line([(5, y), (TERM_W - 5, y)], fill=(51, 255, 51, 3), width=1)

    # Phosphor glow border
    draw.rectangle([4, SCREEN_Y, TERM_W - 4, TERM_H - 4], outline=(51, 255, 51), width=1)

    return img


def make_scanline_overlay(img):
    """Add subtle scanline effect."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(SCREEN_Y, TERM_H - 4, 2):
        draw.line([(5, y), (TERM_W - 5, y)], fill=(0, 0, 0, 12))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def generate_gif(output_path: str):
    font = load_font(FONT_SIZE)
    title_font = load_font(11)

    base = make_base_frame(font, title_font)

    # Curated animation frames from actual Zork 1 gameplay
    # Format: list of (type, text) lines to add, then pause
    script = [
        # Opening
        ([
            ("header",  "══════════════════════════════════════════════════"),
            ("header",  "   AGENTIC ZORK — AI Playing Zork I"),
            ("header",  "   Model: Qwen 3.5 4B (local, Ollama)"),
            ("header",  "══════════════════════════════════════════════════"),
            ("blank",   ""),
            ("game",    "West of House"),
            ("game",    "You are standing in an open field west of a"),
            ("game",    "white house, with a boarded front door."),
            ("game",    "There is a small mailbox here."),
        ], 2500),

        # Step 1: Open mailbox
        ([
            ("blank",   ""),
            ("thought", "💭 Mailboxes require opening. I will open it."),
            ("action",  "▸ open mailbox"),
            ("game",    "Opening the small mailbox reveals a leaflet."),
        ], 2000),

        # Step 2: Examine leaflet
        ([
            ("blank",   ""),
            ("thought", "💭 Examining the leaflet for clues."),
            ("action",  "▸ examine leaflet"),
            ("game",    '"WELCOME TO ZORK! A game of adventure,'),
            ("game",    'danger, and low cunning."'),
        ], 2000),

        # Transition to clearing
        ([
            ("blank",   ""),
            ("dim",     "          ··· exploring forest path ···"),
            ("blank",   ""),
            ("game",    "Clearing"),
            ("game",    "On the ground is a pile of leaves."),
        ], 1500),

        # Step 9: Hidden grating
        ([
            ("blank",   ""),
            ("thought", "💭 Look under leaves might reveal something"),
            ("thought", "   hidden underneath."),
            ("action",  "▸ look under leaves"),
            ("game",    "Underneath the pile of leaves is a grating."),
            ("game",    "As you release the leaves, the grating is"),
            ("game",    "once again concealed from view."),
        ], 2200),

        # Step 10: Valid actions suggest pull
        ([
            ("blank",   ""),
            ("score",   " [Valid actions: pull leaves]"),
            ("thought", "💭 Valid actions suggest pulling the leaves"),
            ("thought", "   to permanently reveal the grating."),
            ("action",  "▸ pull leaves"),
            ("game",    "In disturbing the pile of leaves, a grating"),
            ("game",    "is revealed."),
        ], 2200),

        # Transition to house
        ([
            ("blank",   ""),
            ("dim",     "       ··· navigating to white house ···"),
            ("blank",   ""),
            ("game",    "Behind House"),
            ("game",    "In one corner of the house there is a small"),
            ("game",    "window which is slightly ajar."),
        ], 1500),

        # Step 30: Open window
        ([
            ("blank",   ""),
            ("score",   " [Promising: open window]"),
            ("thought", '💭 The window is "slightly ajar" — opening'),
            ("thought", "   it is the logical action."),
            ("action",  "▸ open window"),
            ("game",    "With great effort, you open the window far"),
            ("game",    "enough to allow entry."),
        ], 2000),

        # Step 31: Enter kitchen (+10!)
        ([
            ("blank",   ""),
            ("thought", "💭 Window is open. Entering to explore the"),
            ("thought", "   house interior."),
            ("action",  "▸ enter"),
            ("game",    "Kitchen"),
            ("game",    "You are in the kitchen of the white house."),
            ("game",    "On the table is an elongated brown sack,"),
            ("game",    "smelling of hot peppers."),
            ("game",    "A bottle is sitting on the table."),
            ("blank",   ""),
            ("reward",  " ★ Score: 10  (+10 points!)"),
        ], 2800),

        # Take items
        ([
            ("blank",   ""),
            ("thought", "💭 Collecting items — the sack and bottle."),
            ("action",  "▸ take elongated brown sack"),
            ("game",    "Taken."),
            ("action",  "▸ take bottle"),
            ("game",    "Taken."),
        ], 1800),

        # Living room
        ([
            ("blank",   ""),
            ("dim",     "         ··· moving to living room ···"),
            ("blank",   ""),
            ("game",    "Living Room"),
            ("game",    "Above the trophy case hangs an elvish sword"),
            ("game",    "of great antiquity."),
            ("game",    "A battery-powered brass lantern is on the"),
            ("game",    "trophy case."),
        ], 1800),

        # Sword error + recovery
        ([
            ("blank",   ""),
            ("thought", "💭 The sword is likely the most valuable item."),
            ("action",  "▸ take elvish sword of great antiquity"),
            ("error",   'I don\'t know the word "great".'),
            ("blank",   ""),
            ("score",   " [Valid actions: take sword, take lantern]"),
            ("thought", '💭 Game rejected "great". Adapting — use'),
            ("thought", "   simpler phrasing from valid actions."),
            ("action",  "▸ take sword"),
            ("game",    "Taken."),
        ], 2500),

        # Lantern
        ([
            ("blank",   ""),
            ("thought", "💭 The lantern provides light for dark areas."),
            ("action",  "▸ take lantern"),
            ("game",    "Taken."),
            ("action",  "▸ turn on lantern"),
            ("game",    "The brass lantern is now on."),
        ], 2200),

        # Final summary
        ([
            ("blank",   ""),
            ("blank",   ""),
            ("header",  "══════════════════════════════════════════════════"),
            ("header",  "   Score: 10 pts | 50 moves | 8 locations"),
            ("header",  "   Items: sword, lantern, sack, bottle"),
            ("header",  "══════════════════════════════════════════════════"),
        ], 4000),
    ]

    frames = []
    durations = []
    accumulated_lines = []

    # Initial empty frame
    empty_frame = render_frame(base, [], font, show_cursor=True)
    empty_frame = make_scanline_overlay(empty_frame)
    frames.append(empty_frame)
    durations.append(800)

    for block_lines, pause_ms in script:
        # Add lines one by one with typing effect
        for i, line in enumerate(block_lines):
            accumulated_lines.append(line)
            frame = render_frame(base, accumulated_lines, font, show_cursor=True)
            frame = make_scanline_overlay(frame)
            frames.append(frame)
            # Fast typing for same block, short delay between lines
            durations.append(120 if line[0] == "blank" else 180)

        # After block: longer pause (reading time)
        # Add frame without cursor for part of the pause (blink effect)
        frame_no_cursor = render_frame(base, accumulated_lines, font, show_cursor=False)
        frame_no_cursor = make_scanline_overlay(frame_no_cursor)
        frame_cursor = render_frame(base, accumulated_lines, font, show_cursor=True)
        frame_cursor = make_scanline_overlay(frame_cursor)

        # Blink cursor during pause
        blinks = max(1, pause_ms // 1000)
        remaining = pause_ms
        for b in range(blinks):
            frames.append(frame_cursor)
            durations.append(min(500, remaining // 2))
            remaining -= min(500, remaining // 2)
            frames.append(frame_no_cursor)
            durations.append(min(500, remaining))
            remaining -= min(500, remaining)
        if remaining > 0:
            frames.append(frame_cursor)
            durations.append(remaining)

    # Save GIF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to palette mode for smaller GIF
    palette_frames = []
    for f in frames:
        pf = f.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
        palette_frames.append(pf)

    palette_frames[0].save(
        output_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,  # Infinite loop
        optimize=True,
    )

    total_ms = sum(durations)
    file_size = Path(output_path).stat().st_size
    print(f"GIF saved to {output_path}")
    print(f"Frames: {len(frames)}, Duration: {total_ms/1000:.1f}s, Size: {file_size/1024:.0f}KB")


if __name__ == "__main__":
    generate_gif("assets/terminal_demo.gif")
