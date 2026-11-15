"""AI腔尺子：Tkinter 离线窗口。"""

import ctypes
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk

from aiqiang import DEFAULT_MODEL_PATH, load_model
from aiqiang.analyzer import MISSING_MODEL_MESSAGE, analyze_loaded, thresholds
from aiqiang.features import cjk_count
from aiqiang.markers import CHECKS, HINTS
from aiqiang.report import about_lines, format_probability, format_report
from aiqiang.textio import read_text

GREY = '#8A8A8A'
VERDICT_COLORS = {'likely_ai': '#D9534F', 'likely_human': '#2E9E5B', 'uncertain': GREY}
CLASS_COLORS = {'DeepSeek': '#4D6BFE', 'GPT': '#10A37F', 'Claude': '#D97757'}
HIGHLIGHT_COLORS = [
    '#FFE0CC', '#FFF0BC', '#DCECCB', '#CDEEE7', '#D8E7FF',
    '#E5DBF5', '#F5D8E8', '#F0E5CE', '#DFE9EA', '#E8E8C9',
]


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def window_layout(scale, screen_width, screen_height):
    """Return physical default/minimum sizes and a centred position."""
    cap_width = int(screen_width * 0.92)
    cap_height = int(screen_height * 0.88)
    size = (min(round(1100 * scale), cap_width), min(round(760 * scale), cap_height))
    minimum = (min(round(900 * scale), cap_width), min(round(600 * scale), cap_height))
    position = ((screen_width - size[0]) // 2, (screen_height - size[1]) // 2)
    return size, minimum, position


class RulerApp:
    def __init__(self, root, model=None, model_path=DEFAULT_MODEL_PATH):
        self.root = root
        self.scale = max(1.0, root.winfo_fpixels('1i') / 96.0)
        self.model = model
        self.model_path = model_path
        self._model_loaded = model is not None
        self.result = None
        self.stale = False
        self._display_text = ''
        self._flash_job = None
        self._copy_job = None
        self._wheel_remainder = 0.0
        self._wheel_tag = 'RulerPanel' + str(id(self))
        self._about_window = None
        self.root.title('AI腔尺子')
        size, minimum, position = window_layout(
            self.scale, root.winfo_screenwidth(), root.winfo_screenheight(),
        )
        self.root.geometry('{}x{}+{}+{}'.format(*(size + position)))
        self.root.minsize(*minimum)
        families = set(font.families(root))
        family = 'Microsoft YaHei UI' if 'Microsoft YaHei UI' in families else None
        default = font.Font(root=root, name='TkDefaultFont', exists=True)
        if family:
            default.configure(family=family)
        default.configure(size=11)
        self.family = default.actual('family')
        self.small_font = font.Font(root=root, family=self.family, size=9)
        self._small_metrics = self.small_font
        self.root.option_add('*Font', default)
        style = ttk.Style(root)
        style.configure('TButton', padding=(self.px(7), self.px(4)), borderwidth=self.px(1))
        style.configure('Vertical.TScrollbar', arrowsize=self.px(14), width=self.px(14))
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.root.bind_class(self._wheel_tag, sequence, self._on_panel_wheel)
        self._build()
        self.root.bind('<Control-Return>', self.analyze_text)
        self.root.bind('<Control-KP_Enter>', self.analyze_text)
        self._update_count()

    def px(self, value):
        return round(value * self.scale)

    def _build(self):
        px = self.px
        outer = ttk.Frame(self.root, padding=px(12))
        outer.pack(fill='both', expand=True)
        outer.columnconfigure(0, weight=3, uniform='columns')
        outer.columnconfigure(1, weight=2, uniform='columns')
        outer.rowconfigure(0, weight=1)
        left = ttk.Frame(outer, padding=(0, 0, px(12), 0))
        self.left_panel = left
        left.grid(row=0, column=0, sticky='nsew')
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        self.input_prompt = ttk.Label(
            left, text='把中文粘贴到下面（建议 100 字以上）', wraplength=px(450),
        )
        self.input_prompt.grid(
            row=0, column=0, sticky='ew', pady=(0, px(8)),
        )
        left.bind('<Configure>', self._resize_left)
        editor = ttk.Frame(left)
        editor.grid(row=1, column=0, sticky='nsew')
        editor.rowconfigure(0, weight=1)
        editor.columnconfigure(0, weight=1)
        self.input_text = tk.Text(
            editor, wrap='char', undo=True, font=(self.family, 11),
            width=1, height=1, padx=px(10), pady=px(10), relief='solid',
            borderwidth=px(1), spacing3=px(4), highlightthickness=0,
        )
        self.input_text.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(editor, orient='vertical', command=self.input_text.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        self.input_text.configure(yscrollcommand=scroll.set)
        self.input_text.bind('<<Modified>>', self._on_edit)
        buttons = ttk.Frame(left)
        buttons.grid(row=2, column=0, sticky='ew', pady=(px(10), px(4)))
        for column in range(3):
            buttons.columnconfigure(column, weight=1, uniform='buttons')
        self.buttons = []
        for i, (label, command) in enumerate([
            ('量一量', self.analyze_text), ('打开文件…', self.open_file),
            ('清空', self.clear), ('示例', self.load_sample), ('复制结果', self.copy_result),
        ]):
            button = ttk.Button(buttons, text=label, command=command, width=1)
            button.grid(
                row=i // 3, column=i % 3, columnspan=2 if i == 4 else 1,
                sticky='ew', padx=px(3), pady=px(2),
            )
            self.buttons.append(button)
        self.copy_button = self.buttons[-1]
        self.copy_button.state(['disabled'])
        self.status = tk.StringVar(master=self.root)
        self.status_label = ttk.Label(left, textvariable=self.status, foreground=GREY, wraplength=px(450))
        self.status_label.grid(
            row=3, column=0, sticky='w', pady=(px(4), 0),
        )
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        background = ttk.Style(self.root).lookup('TFrame', 'background') or '#F0F0F0'
        self.right_canvas = tk.Canvas(
            right, width=px(320), height=px(1), highlightthickness=0, borderwidth=0, background=background,
        )
        self.right_canvas.grid(row=0, column=0, sticky='nsew')
        right_scroll = ttk.Scrollbar(right, orient='vertical', command=self.right_canvas.yview)
        right_scroll.grid(row=0, column=1, sticky='ns')
        self.right_canvas.configure(yscrollcommand=right_scroll.set)
        panel = ttk.Frame(self.right_canvas, padding=(0, 0, px(5), 0))
        self.right_panel = panel
        self._panel_id = self.right_canvas.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', self._resize_panel)
        self.right_canvas.bind('<Configure>', self._resize_canvas)
        panel.columnconfigure(0, weight=1)
        header = ttk.Frame(panel)
        header.grid(row=0, column=0, sticky='ew')
        ttk.Label(header, text='文风测量', font=(self.family, 11, 'bold')).pack(side='left')
        self.about_button = ttk.Button(header, text='关于', command=self.show_about)
        self.about_button.pack(side='right')
        self.verdict_label = ttk.Label(
            panel, text='粘贴文字，量一量', foreground=GREY,
            font=(self.family, 19, 'bold'), wraplength=px(360),
        )
        self.verdict_label.grid(row=1, column=0, sticky='w', pady=(px(6), px(8)))
        probability = ttk.Frame(panel)
        probability.grid(row=2, column=0, sticky='ew')
        self.prob_title = ttk.Label(probability, text='AI 可能性', foreground=GREY)
        self.prob_title.pack(side='left')
        self.prob_value = ttk.Label(probability, text='—', font=(self.family, 23, 'bold'), foreground=GREY)
        self.prob_value.pack(side='left', padx=(px(10), 0))
        self.prob_canvas = tk.Canvas(
            panel, width=px(300), height=px(42) + self._small_metrics.metrics('linespace'),
            highlightthickness=0, borderwidth=0, background=background,
        )
        self.prob_canvas.grid(row=3, column=0, sticky='ew', pady=(0, px(6)))
        self.prob_canvas.bind('<Configure>', lambda event: self._draw_probability())
        self.pieces_frame = ttk.Frame(panel, width=1, height=1)
        self.pieces_frame.grid(row=4, column=0, sticky='ew', pady=(0, px(6)))
        self.pieces_frame.grid_remove()
        self.pieces_frame.bind('<Configure>', lambda event: self._layout_pieces())
        self.which_title = ttk.Label(panel, text='更像哪个 AI', foreground=GREY)
        self.which_title.grid(row=5, column=0, sticky='w')
        self.which_frame = ttk.Frame(panel)
        self.which_frame.grid(row=6, column=0, sticky='ew')
        self.which_frame.columnconfigure(1, weight=1)
        self.which_rows = {}
        self.which_label = ttk.Label(panel, foreground=GREY, wraplength=px(360), font=self.small_font)
        self.which_label.grid(row=7, column=0, sticky='w', pady=(px(2), px(8)))
        self.baigu_label = ttk.Label(panel, text='AI 八股 0/10', font=(self.family, 11, 'bold'))
        self.baigu_label.grid(row=8, column=0, sticky='w', pady=(0, px(4)))
        self.marker_rows = []
        for i, ((name, _), hint) in enumerate(zip(CHECKS, HINTS)):
            row = ttk.Frame(panel)
            row.grid(row=9 + i, column=0, sticky='ew', pady=px(2))
            row.columnconfigure(2, weight=1)
            square = tk.Canvas(
                row, width=px(12), height=px(12), background='#E5E5E5', borderwidth=0, highlightthickness=0,
            )
            square.grid(row=0, column=0, padx=(0, px(6)))
            label = ttk.Label(row, text=name + ' ×0', foreground=GREY)
            label.grid(row=0, column=1, sticky='w', padx=(0, px(6)))
            detail = ttk.Label(row, text=hint, foreground=GREY, font=self.small_font, wraplength=px(300))
            detail.grid(row=1, column=1, columnspan=2, sticky='w')
            row.bind('<Configure>', lambda event, index=i: self._layout_marker(index))
            for widget in (row, square, label, detail):
                widget.bind('<Button-1>', lambda event, index=i: self.jump_to_marker(index))
            self.marker_rows.append((square, label, detail))
        self.evidence_frames = {}
        self.evidence_titles = {}
        for row, key, label in [(19, 'ai', '像 AI 的写法'), (21, 'human', '像人的写法')]:
            title = ttk.Label(panel, text=label, font=self.small_font, foreground=GREY)
            title.grid(row=row, column=0, sticky='w', pady=(px(7), px(2)))
            self.evidence_titles[key] = title
            frame = ttk.Frame(panel)
            frame.grid(row=row + 1, column=0, sticky='ew')
            for column in range(2):
                frame.columnconfigure(column, weight=1, uniform='evidence')
            frame.bind('<Configure>', self._resize_evidence)
            self.evidence_frames[key] = frame
        footer = (
            '只量文风，不下结论。别拿它去指控任何人。'
            '离线运行，不上传任何内容。'
        )
        self.footer = ttk.Label(panel, text=footer, foreground=GREY, font=self.small_font, wraplength=px(360))
        self.footer.grid(row=23, column=0, sticky='w', pady=(px(10), px(2)))
        self.notes_label = ttk.Label(panel, foreground=GREY, font=self.small_font, wraplength=px(360))
        self.notes_label.grid(row=24, column=0, sticky='w', pady=(px(3), px(4)))
        self._build_which_rows()
        self._bind_panel_wheel(right)

    def _resize_left(self, event):
        wrap = max(1, event.width - self.px(16))
        self.input_prompt.configure(wraplength=wrap)
        self.status_label.configure(wraplength=wrap)

    def _resize_panel(self, event=None):
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox('all'))

    def _resize_canvas(self, event):
        self.right_canvas.itemconfigure(self._panel_id, width=event.width)
        wrap = max(1, event.width - self.px(12))
        for widget in (self.verdict_label, self.which_label, self.footer, self.notes_label):
            widget.configure(wraplength=wrap)

    def _layout_marker(self, index):
        square, label, hint = self.marker_rows[index]
        row = label.master
        available = max(1, row.winfo_width() - square.winfo_reqwidth() - self.px(6))
        inline = available - label.winfo_reqwidth() - self.px(6)
        if inline >= self._small_metrics.measure(hint.cget('text')) + self.px(4):
            hint.grid_configure(row=0, column=2, columnspan=1)
            hint.configure(wraplength=inline)
        else:
            hint.grid_configure(row=1, column=1, columnspan=2)
            hint.configure(wraplength=max(1, available - self.px(6)))

    def _resize_evidence(self, event):
        wrap = max(1, event.width // 2 - self.px(10))
        for child in event.widget.winfo_children():
            child.configure(wraplength=wrap)

    def _layout_pieces(self):
        frame = self.pieces_frame
        width = max(1, frame.winfo_width())
        x, y, height = 0, 0, 0
        gap = self.px(5)
        for chip in frame.winfo_children():
            chip_width = min(width, chip.winfo_reqwidth())
            chip_height = chip.winfo_reqheight()
            if x and x + chip_width > width:
                x, y, height = 0, y + height + gap, 0
            chip.place(x=x, y=y, width=chip_width, height=chip_height)
            x += chip_width + gap
            height = max(height, chip_height)
        frame.configure(height=y + height)

    def _render_pieces(self):
        frame = self.pieces_frame
        for child in frame.winfo_children():
            child.destroy()
        pieces = self.result['pieces']
        if self.stale or len(pieces) <= 1:
            frame.grid_remove()
            return
        limits = thresholds(self.model)
        for i, piece in enumerate(pieces):
            p = piece['ai_prob']
            if p >= limits['ai_high']:
                color = VERDICT_COLORS['likely_ai']
            elif p <= limits['human_low']:
                color = VERDICT_COLORS['likely_human']
            else:
                color = GREY
            tk.Button(
                frame, text='第{}段 {}'.format(i + 1, format_probability(p)),
                font=self.small_font, foreground=color, activeforeground=color,
                relief='solid', borderwidth=self.px(1), cursor='hand2',
                padx=self.px(6), pady=self.px(2), highlightthickness=0,
                command=lambda index=i: self.jump_to_piece(index),
            )
        frame.grid()
        self._layout_pieces()
        self._bind_panel_wheel(frame)

    def _bind_panel_wheel(self, widget):
        tags = widget.bindtags()
        if self._wheel_tag not in tags:
            widget.bindtags((tags[0], self._wheel_tag) + tags[1:])
        for child in widget.winfo_children():
            self._bind_panel_wheel(child)

    def _on_panel_wheel(self, event):
        first, last = self.right_canvas.yview()
        if first <= 0 and last >= 1:
            return
        if event.num in (4, 5):
            steps = -1 if event.num == 4 else 1
        else:
            self._wheel_remainder -= event.delta / 120.0
            steps = int(self._wheel_remainder)
            self._wheel_remainder -= steps
        self.right_canvas.yview_scroll(steps, 'units')
        return 'break'

    def _build_which_rows(self):
        classes = self.model['classes'][1:] if self.model else ['DeepSeek', 'GPT', 'Claude']
        if list(self.which_rows) == classes:
            return
        for child in self.which_frame.winfo_children():
            child.destroy()
        self.which_rows = {}
        for i, name in enumerate(classes):
            label = ttk.Label(self.which_frame, text=name, foreground=GREY, font=self.small_font)
            label.grid(row=i, column=0, sticky='w', padx=(0, self.px(6)))
            canvas = tk.Canvas(
                self.which_frame, height=self.px(15), width=self.px(100), highlightthickness=0, borderwidth=0,
            )
            canvas.grid(row=i, column=1, sticky='ew', pady=self.px(2))
            percent = ttk.Label(self.which_frame, text='—', foreground=GREY, font=self.small_font, width=8)
            percent.grid(row=i, column=2, sticky='e', padx=(self.px(6), 0))
            canvas.bind('<Configure>', lambda event: self._draw_which())
            self.which_rows[name] = (label, canvas, percent)
        self._bind_panel_wheel(self.which_frame)

    def _ensure_model(self):
        if not self._model_loaded:
            self.status.set('正在加载模型…')
            self.root.update_idletasks()
            self.model = load_model(self.model_path)
            self._model_loaded = True
            self._build_which_rows()

    def _update_count(self):
        if self._copy_job is not None:
            self.root.after_cancel(self._copy_job)
            self._copy_job = None
        text = self.input_text.get('1.0', 'end-1c')
        status = '{} 个汉字'.format(cjk_count(text))
        if self.result and self.result['verdict'] == 'no_model' and not self.stale:
            status += ' ｜ ' + MISSING_MODEL_MESSAGE
        self.status.set(status)

    def _clear_highlights(self):
        if self._flash_job is not None:
            self.root.after_cancel(self._flash_job)
            self._flash_job = None
        self.input_text.tag_remove('sel', '1.0', 'end')
        for tag in self.input_text.tag_names():
            if tag.startswith(('marker_', 'evidence_')):
                self.input_text.tag_delete(tag)

    def _on_edit(self, event=None):
        if not self.input_text.edit_modified():
            return
        self.input_text.edit_modified(False)
        if self.result is not None:
            self.stale = True
            self._clear_highlights()
            self._render_result()
        self._update_count()

    def _text_index(self, offset):
        # Tcl 8.6 may count an emoji as two columns. Keep Python offsets in the API.
        prefix = self._display_text[:offset]
        line = prefix.count('\n') + 1
        column = self.root.tk.call('string', 'length', prefix.rsplit('\n', 1)[-1])
        return '{}.{}'.format(line, column)

    def _highlight(self):
        self._clear_highlights()
        for i, row in enumerate(self.result['markers']):
            tag = 'marker_{}'.format(i)
            self.input_text.tag_configure(tag, background=HIGHLIGHT_COLORS[i])
            for start, end in row['spans']:
                self.input_text.tag_add(tag, self._text_index(start), self._text_index(end))
        for kind, color in [('ai', '#D67A17'), ('human', '#2E9E5B')]:
            tag = 'evidence_' + kind
            self.input_text.tag_configure(tag, underline=True, foreground=color)
            for gram, _ in self.result['evidence'][kind]:
                start = self._display_text.find(gram)
                while start != -1:
                    end = start + len(gram)
                    self.input_text.tag_add(tag, self._text_index(start), self._text_index(end))
                    start = self._display_text.find(gram, start + 1)

    def analyze_text(self, event=None):
        try:
            self._ensure_model()
            self._display_text = self.input_text.get('1.0', 'end-1c')
            self.input_text.edit_modified(False)
            self.result = analyze_loaded(self._display_text, self.model)
            self.stale = False
            self._render_result()
            self._highlight()
            self._update_count()
        except Exception as error:
            self._update_count()
            messagebox.showerror('无法完成测量', str(error), parent=self.root)
        return 'break'

    def _render_result(self):
        result = self.result
        label = '文字改动了，再量一次' if self.stale else result['verdict_text']
        color = GREY if self.stale else VERDICT_COLORS.get(result['verdict'], GREY)
        self.verdict_label.configure(text=label, foreground=color)
        has_probability = result['ai_prob'] is not None and not self.stale
        self.prob_title.configure(foreground='#333333' if has_probability else GREY)
        self.prob_value.configure(
            text=format_probability(result['ai_prob']) if has_probability else '—', foreground=color,
        )
        self.copy_button.state(['disabled'] if self.stale else ['!disabled'])
        self.which_label.configure(text=result['which_text'], foreground=GREY)
        self.baigu_label.configure(
            text='AI 八股 {}/10'.format(result['baigu_index']), foreground=GREY if self.stale else '#333333',
        )
        for i, row in enumerate(result['markers']):
            square, label, hint = self.marker_rows[i]
            active = row['count'] > 0 and not self.stale
            label.configure(text='{} ×{}'.format(row['name'], row['count']), foreground='#333333' if active else GREY)
            square.configure(background=HIGHLIGHT_COLORS[i] if active else '#E5E5E5')
            hint.configure(foreground=GREY)
            self._layout_marker(i)
        for key, frame in self.evidence_frames.items():
            self.evidence_titles[key].configure(foreground=GREY if self.stale else '#333333')
            for child in frame.winfo_children():
                child.destroy()
            grams = result['evidence'][key]
            if not grams:
                ttk.Label(frame, text='暂无', foreground=GREY, font=self.small_font).grid(row=0, column=0)
            for i, (gram, _) in enumerate(grams):
                color = GREY if self.stale else ('#B66A10' if key == 'ai' else '#278548')
                ttk.Label(
                    frame, text=gram, foreground=color, font=self.small_font,
                    wraplength=max(1, frame.winfo_width() // 2 - self.px(10)),
                ).grid(
                    row=i // 2, column=i % 2, sticky='w', padx=(0, self.px(5)),
                )
            self._bind_panel_wheel(frame)
        self.notes_label.configure(text='\n'.join(result['notes'][2:]))
        self._render_pieces()
        self._draw_probability()
        self._draw_which()

    def _draw_probability(self):
        canvas = self.prob_canvas
        canvas.delete('all')
        if not self.result or self.result['ai_prob'] is None:
            return
        px = self.px
        width = canvas.winfo_width()
        if width <= px(12):
            return
        left, right = px(4), width - px(4)
        limits = thresholds(self.model)
        low, high = limits['human_low'], limits['ai_high']
        zones = [
            (0, low, '#D7EEDF', '更像人'), (low, high, '#E7E7E7', '不确定'),
            (high, 1, '#F5D9D8', '很像 AI'),
        ]
        captions = []
        for a, b, color, caption in zones:
            start, end = left + a * (right - left), left + b * (right - left)
            canvas.create_rectangle(
                start, px(12), end, px(26), fill='#E7E7E7' if self.stale else color,
                outline='', tags='zone',
            )
            text_width = self._small_metrics.measure(caption) + px(4)
            if end - start >= text_width:
                x = (start + end - text_width) / 2
            else:
                x = start if a + b <= 1 else end - text_width
            captions.append([max(left, min(x, right - text_width)), text_width, caption])
        # Keep edge captions readable even when custom thresholds leave tiny zones.
        for previous, current in zip(captions, captions[1:]):
            current[0] = max(current[0], previous[0] + previous[1] + px(4))
        captions[-1][0] = min(captions[-1][0], right - captions[-1][1])
        for i in range(len(captions) - 2, -1, -1):
            following = captions[i + 1]
            captions[i][0] = min(captions[i][0], following[0] - px(4) - captions[i][1])
        for x, _, caption in captions:
            canvas.create_text(
                x + px(2), px(34), text=caption, anchor='nw', fill=GREY,
                font=self.small_font, tags='zone_caption',
            )
        p = self.result['ai_prob']
        x = left + p * (right - left)
        color = GREY if self.stale else VERDICT_COLORS.get(self.result['verdict'], GREY)
        canvas.create_polygon(
            x - px(4), px(2), x + px(4), px(2), x, px(10),
            fill=color, outline='', tags='probability_triangle',
        )
        canvas.create_line(x, px(10), x, px(29), width=px(1), fill=color, tags='probability_line')

    def _draw_which(self):
        inactive = not self.result or self.stale or self.result['verdict'] in (
            'likely_human', 'too_short', 'no_model',
        )
        self.which_title.configure(foreground=GREY if inactive else '#333333')
        scores = {row['name']: row['p'] for row in self.result['which']} if self.result else {}
        for name, (label, canvas, percent) in self.which_rows.items():
            color = GREY if inactive else CLASS_COLORS.get(name, GREY)
            label.configure(foreground=color)
            percent.configure(text=format_probability(scores[name]) if name in scores else '—', foreground=color)
            canvas.delete('all')
            width = max(1, canvas.winfo_width())
            canvas.create_rectangle(0, self.px(2), width, self.px(13), fill='#E7E7E7', outline='')
            if name in scores:
                canvas.create_rectangle(
                    0, self.px(2), width * scores[name], self.px(13), fill=color, outline='',
                )

    def copy_result(self):
        if self.result is None or self.stale:
            return
        if self.input_text.get('1.0', 'end-1c') != self._display_text:
            self._on_edit()
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(format_report(self.result, self._display_text))
            self._update_count()
            self.status.set('已复制')
            self._copy_job = self.root.after(2000, self._update_count)
        except tk.TclError as error:
            messagebox.showerror('无法复制结果', str(error), parent=self.root)

    def jump_to_marker(self, index):
        if not self.result or self.stale:
            return
        spans = self.result['markers'][index]['spans']
        if not spans:
            return
        self._select_span(*spans[0], brief=True)

    def jump_to_piece(self, index):
        if not self.result or self.stale:
            return
        piece = self.result['pieces'][index]
        self._select_span(piece['start'], piece['end'])

    def _select_span(self, start, end, brief=False):
        if self._flash_job is not None:
            self.root.after_cancel(self._flash_job)
        self._clear_selection()
        a, b = self._text_index(start), self._text_index(end)
        self.input_text.see(a)
        self.input_text.tag_add('sel', a, b)
        self.input_text.focus_set()
        if brief:
            self._flash_job = self.root.after(1400, self._clear_selection)

    def _clear_selection(self):
        self.input_text.tag_remove('sel', '1.0', 'end')
        self._flash_job = None

    def _replace_text(self, text):
        self.input_text.delete('1.0', 'end')
        self.input_text.insert('1.0', text)
        self._on_edit()
        self.input_text.edit_reset()
        self.input_text.focus_set()

    def open_file(self):
        path = filedialog.askopenfilename(
            parent=self.root, title='打开文字文件', filetypes=[('文字文件', '*.txt'), ('所有文件', '*')],
        )
        if path:
            try:
                self._replace_text(read_text(path))
            except Exception as error:
                messagebox.showerror('无法打开文件', str(error), parent=self.root)

    def clear(self):
        self._replace_text('')

    def load_sample(self):
        try:
            self._replace_text(read_text(Path(__file__).resolve().parent / 'aiqiang' / 'sample.txt'))
        except Exception as error:
            messagebox.showerror('无法打开示例', str(error), parent=self.root)

    def show_about(self):
        try:
            self._ensure_model()
            self._update_count()
            if self._about_window is not None and self._about_window.winfo_exists():
                self._about_window.lift()
                return
            window = tk.Toplevel(self.root)
            self._about_window = window
            window.title('关于 AI腔尺子')
            cap_width = int(window.winfo_screenwidth() * 0.92)
            cap_height = int(window.winfo_screenheight() * 0.88)
            width, height = min(self.px(600), cap_width), min(self.px(420), cap_height)
            x = (window.winfo_screenwidth() - width) // 2
            y = (window.winfo_screenheight() - height) // 2
            window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
            window.minsize(min(self.px(440), cap_width), min(self.px(280), cap_height))
            window.transient(self.root)
            frame = ttk.Frame(window, padding=self.px(16))
            frame.pack(fill='both', expand=True)
            text = tk.Text(
                frame, wrap='char', font=(self.family, 11), relief='flat',
                padx=self.px(8), pady=self.px(8), borderwidth=0, highlightthickness=0,
            )
            scrollbar = ttk.Scrollbar(frame, command=text.yview)
            scrollbar.pack(side='right', fill='y')
            text.configure(yscrollcommand=scrollbar.set)
            text.pack(fill='both', expand=True)
            text.insert('1.0', '\n\n'.join(about_lines(self.model)))
            text.configure(state='disabled')
            ttk.Button(window, text='关闭', command=window.destroy).pack(pady=(0, self.px(12)))
        except Exception as error:
            messagebox.showerror('无法显示关于', str(error), parent=self.root)


def main():
    enable_dpi_awareness()
    root = tk.Tk()
    RulerApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
