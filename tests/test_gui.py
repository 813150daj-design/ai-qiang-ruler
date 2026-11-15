import os
import tkinter as tk
import unittest
from unittest.mock import patch

import ruler_gui
from aiqiang import __version__
from aiqiang.analyzer import VERDICT_LABELS, thresholds, verdict_for
from aiqiang.report import format_probability, format_report
from common import MODEL_PATH, ROOT, read_json


@unittest.skipIf('AIQ_SKIP_GUI' in os.environ, 'AIQ_SKIP_GUI is set')
class GUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ruler_gui.enable_dpi_awareness()

    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest('Tk cannot start: ' + str(error))
        self.addCleanup(self.root.destroy)
        self.root.withdraw()
        self.app = ruler_gui.RulerApp(self.root, model=read_json(MODEL_PATH))
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        self.popup = patch.object(ruler_gui.messagebox, 'showerror')
        self.showerror = self.popup.start()
        self.addCleanup(self.popup.stop)

    def analyze_sample(self):
        self.app.load_sample()
        self.root.update()
        self.app.analyze_text()
        self.root.update()
        self.showerror.assert_not_called()
        self.assertEqual(self.errors, [])

    def test_analyze_highlights_jump_and_edit_invalidation(self):
        self.analyze_sample()
        self.assertEqual(self.app.verdict_label.cget('text'), self.app.result['verdict_text'])
        self.assertTrue(any(
            self.app.input_text.tag_ranges(tag)
            for tag in self.app.input_text.tag_names() if tag.startswith('marker_')
        ))
        self.app.jump_to_marker(0)
        self.assertTrue(self.app.input_text.tag_ranges('sel'))
        self.app.input_text.insert('end', '又添了一句话。')
        self.root.update()
        self.assertEqual(self.app.verdict_label.cget('text'), '文字改动了，再量一次')
        self.assertFalse(any(tag.startswith('marker_') for tag in self.app.input_text.tag_names()))
        self.assertFalse(any(tag.startswith('evidence_') for tag in self.app.input_text.tag_names()))
        self.assertEqual(self.errors, [])

    def test_unicode_offsets_and_about_summary(self):
        text = '😀。\n  我来，我看，我想，我走。\n人生很温柔。' + '中' * 60
        self.app._replace_text(text)
        self.root.update()
        self.app.analyze_text()
        self.root.update()
        for i, row in enumerate(self.app.result['markers']):
            ranges = self.app.input_text.tag_ranges('marker_{}'.format(i))
            for j, (start, end) in enumerate(row['spans']):
                highlighted = self.app.input_text.get(ranges[j * 2], ranges[j * 2 + 1])
                self.assertEqual(highlighted, text[start:end])
        self.app.show_about()
        self.root.update()
        window = self.app._about_window
        self.assertTrue(window.winfo_exists())
        texts = [w for frame in window.winfo_children() for w in frame.winfo_children() if isinstance(w, tk.Text)]
        content = texts[0].get('1.0', 'end-1c')
        for line in self.app.model['eval']['summary_lines']:
            self.assertIn(line, content)
        self.assertIn('代码 MIT；模型权重 CC BY-SA 4.0（训练数据含维基百科）', content)
        self.showerror.assert_not_called()
        self.assertEqual(self.errors, [])

    def test_piece_chips_select_unicode_offsets_colors_and_edit_invalidation(self):
        self.root.deiconify()
        self.root.geometry('{}x{}'.format(*self.root.minsize()))
        text = '😀。\r\n' + ('中' * 149 + '。\r\n\n') * 12 + ' \r\n'
        self.app._replace_text(text)
        self.root.update()
        self.app.analyze_text()
        self.root.update()
        pieces = self.app.result['pieces']
        chips = self.app.pieces_frame.winfo_children()
        self.assertEqual(len(chips), len(pieces))
        self.assertGreaterEqual(len(chips), 6)
        for i, (chip, piece) in enumerate(zip(chips, pieces), 1):
            self.assertEqual(chip.cget('text'), '第{}段 {}'.format(i, format_probability(piece['ai_prob'])))
            chip.invoke()
            self.root.update()
            self.assertEqual(self.app.input_text.get('sel.first', 'sel.last'), text[piece['start']:piece['end']])
            self.assertIsNotNone(self.app.input_text.bbox(self.app._text_index(piece['start'])))
        self.assertGreater(self.app.input_text.yview()[0], 0)
        self.app.model['thresholds'].update(human_low=0.3, ai_high=0.8)
        probabilities = (0.0, 0.3, 0.300001, 0.799999, 0.8, 1.0)
        for piece, p in zip(pieces, probabilities):
            piece['ai_prob'] = p
        self.app._render_result()
        self.root.update()
        colors = [str(chip.cget('foreground')) for chip in self.app.pieces_frame.winfo_children()[:6]]
        self.assertEqual(colors, ['#2E9E5B', '#2E9E5B', ruler_gui.GREY, ruler_gui.GREY, '#D9534F', '#D9534F'])
        self.app.copy_button.invoke()
        self.assertIn('分段：第1段 1% 以下 ｜ 第2段 30%', self.root.clipboard_get())
        self.app.input_text.insert('end', '又写一句。')
        self.root.update()
        self.assertEqual(self.app.pieces_frame.winfo_children(), [])
        self.assertFalse(self.app.pieces_frame.winfo_ismapped())
        self.assertFalse(self.app.input_text.tag_ranges('sel'))
        self.app._replace_text('中。' * 60)
        self.root.update()
        self.app.analyze_text()
        self.root.update()
        self.assertEqual(len(self.app.result['pieces']), 1)
        self.assertEqual(self.app.pieces_frame.winfo_children(), [])
        self.assertFalse(self.app.pieces_frame.winfo_ismapped())
        self.showerror.assert_not_called()
        self.assertEqual(self.errors, [])

    def test_lazy_missing_model_still_highlights(self):
        other = tk.Toplevel(self.root)
        self.addCleanup(other.destroy)
        app = ruler_gui.RulerApp(other, model_path=ROOT / 'tests' / 'missing-model.json')
        self.assertFalse(app._model_loaded)
        app._replace_text('原来——这一刻真温柔。')
        other.update()
        app.analyze_text()
        other.update()
        self.assertEqual(app.result['verdict'], 'no_model')
        self.assertEqual(app.prob_value.cget('text'), '—')
        self.assertIn('还没有模型文件 aiqiang/model.json', app.status.get())
        self.assertTrue(app.input_text.tag_ranges('marker_2'))
        self.showerror.assert_not_called()
        self.assertEqual(self.errors, [])

    def test_copy_report_expiry_and_edit_invalidation(self):
        self.assertTrue(self.app.copy_button.instate(['disabled']))
        self.analyze_sample()
        self.assertTrue(self.app.copy_button.instate(['!disabled']))
        self.app.copy_button.invoke()
        self.assertEqual(self.root.clipboard_get(), format_report(self.app.result, self.app._display_text))
        self.assertEqual(self.app.status.get(), '已复制')
        wait = tk.BooleanVar(self.root)
        self.root.after(2100, lambda: wait.set(True))
        self.root.wait_variable(wait)
        self.assertIsNone(self.app._copy_job)
        self.assertIn('个汉字', self.app.status.get())
        self.app.copy_button.invoke()
        self.app.input_text.insert('end', '改动。')
        self.root.update()
        self.assertTrue(self.app.copy_button.instate(['disabled']))
        self.assertIsNone(self.app._copy_job)
        self.assertNotEqual(self.app.status.get(), '已复制')
        self.assertEqual(self.errors, [])

    def test_probability_limits_in_window_and_clipboard(self):
        self.analyze_sample()
        self.addCleanup(self.app._update_count)
        for row, probability in zip(self.app.result['which'], (1.0, 0.0, 0.0)):
            row['p'] = probability
        for probability, display in [(0.0, '1% 以下'), (0.01, '1% 以下'), (0.99, '99%+'), (1.0, '99%+')]:
            with self.subTest(probability=probability):
                self.app.result['ai_prob'] = probability
                self.app._render_result()
                self.root.update()
                self.assertEqual(self.app.prob_value.cget('text'), display)
                for row, expected in zip(self.app.result['which'], ('99%+', '1% 以下', '1% 以下')):
                    percent = self.app.which_rows[row['name']][2]
                    self.assertEqual(percent.cget('text'), expected)
                self.app.copy_button.invoke()
                report = self.root.clipboard_get()
                self.assertIn('AI 可能性 ' + display, report)
                for row, expected in zip(self.app.result['which'], ('99%+', '1% 以下', '1% 以下')):
                    self.assertIn(row['name'] + ' ' + expected, report)
                self.assertEqual(self.app.result['ai_prob'], probability)
        self.showerror.assert_not_called()
        self.assertEqual(self.errors, [])

    def test_five_tier_colors_notes_copy_and_inactive_human_panel(self):
        self.root.deiconify()
        self.app.model['thresholds'].update(mid_high=0.8, mid_low=0.5)
        self.app.model['tiers'] = {
            code: dict(label=label, note='本档实测说明，随面板宽度自动换行。' * 4)
            for code, label in VERDICT_LABELS.items()
        }
        text = '我今天去街口买菜，回来发现下雨了。' * 8
        self.app._replace_text(text)
        self.root.update()
        for p, code, color in [
            (0.95, 'likely_ai', '#D9534F'), (0.85, 'leans_ai', '#E8893A'),
            (0.6, 'uncertain', ruler_gui.GREY), (0.4, 'leans_human', '#6DBF8B'),
            (0.1, 'likely_human', '#2E9E5B'),
        ]:
            prediction = dict(
                ai_prob=p, probs=[0.1, 0.6, 0.2, 0.1], which=[0.7, 0.2, 0.1],
                pieces=[[0, len(text) // 2, p], [len(text) // 2, len(text), p]],
            )
            with self.subTest(code=code):
                with patch('aiqiang.analyzer.predict_doc', return_value=prediction):
                    self.app.analyze_text()
                self.root.update()
                self.assertEqual(str(self.app.verdict_label.cget('foreground')), color)
                self.assertEqual(self.app.verdict_note.cget('text'), self.app.result['verdict_note'])
                self.assertEqual(str(self.app.verdict_note.cget('foreground')), ruler_gui.GREY)
                self.assertTrue(self.app.verdict_note.winfo_ismapped())
                self.assertLessEqual(
                    self.app.verdict_note.winfo_width(), self.app.verdict_note.master.winfo_width(),
                )
                self.assertGreater(
                    self.app.verdict_note.winfo_height(), self.app.small_font.metrics('linespace'),
                )
                for tag in ('probability_triangle', 'probability_line'):
                    self.assertEqual(self.app.prob_canvas.itemcget(tag, 'fill'), color)
                for chip in self.app.pieces_frame.winfo_children():
                    self.assertEqual(str(chip.cget('foreground')), color)
                if code in ('likely_human', 'leans_human'):
                    self.assertEqual(str(self.app.which_title.cget('foreground')), ruler_gui.GREY)
                    for label, canvas, percent in self.app.which_rows.values():
                        self.assertEqual(str(label.cget('foreground')), ruler_gui.GREY)
                        self.assertEqual(str(percent.cget('foreground')), ruler_gui.GREY)
                        self.assertEqual(canvas.itemcget(canvas.find_all()[-1], 'fill'), ruler_gui.GREY)
                self.app.copy_button.invoke()
                self.assertIn('说明：' + self.app.result['verdict_note'], self.root.clipboard_get())
        self.assertEqual(self.root.title(), 'AI腔尺子 ' + __version__)
        self.app.input_text.insert('end', '改动。')
        self.root.update()
        self.assertFalse(self.app.verdict_note.winfo_ismapped())
        self.showerror.assert_not_called()
        self.assertEqual(self.errors, [])

    def test_equal_segments_captions_and_interpolated_marker_at_three_widths(self):
        self.root.deiconify()
        self.root.minsize(1, 1)
        self.analyze_sample()
        base = dict(min_chars=50, ai_high=0.9, human_low=0.2, which_min=0.5)
        for mids, expected in [
            ({}, [
                ('更像人', '#D7EEDF', 0.0, 0.2),
                ('不确定', '#E7E7E7', 0.2, 0.9),
                ('很像 AI', '#F5D9D8', 0.9, 1.0),
            ]),
            (dict(mid_low=0.5, mid_high=0.8), [
                ('更像人', '#D7EEDF', 0.0, 0.2),
                ('偏像人', '#E6F3EA', 0.2, 0.5),
                ('不确定', '#E7E7E7', 0.5, 0.8),
                ('偏像 AI', '#F9E6D3', 0.8, 0.9),
                ('很像 AI', '#F5D9D8', 0.9, 1.0),
            ]),
        ]:
            self.app.model['thresholds'] = dict(base, **mids)
            for width in (600, 900, 1200):
                with self.subTest(mids=mids, width=width):
                    self.root.geometry('{}x{}'.format(self.app.px(width), self.app.px(700)))
                    self.root.update()
                    self.app._draw_probability()
                    self.root.update()
                    canvas = self.app.prob_canvas
                    captions = canvas.find_withtag('zone_caption')
                    self.assertEqual([canvas.itemcget(c, 'text') for c in captions], [e[0] for e in expected])
                    boxes = [canvas.bbox(c) for c in captions]
                    for i, box in enumerate(boxes):
                        self.assertGreaterEqual(box[0], 0)
                        self.assertLessEqual(box[2], canvas.winfo_width())
                        self.assertLessEqual(box[3], canvas.winfo_height())
                        for other in boxes[:i]:
                            self.assertTrue(
                                box[2] < other[0] or other[2] < box[0]
                                or box[3] < other[1] or other[3] < box[1],
                                (box, other),
                            )
                    zones = canvas.find_withtag('zone')
                    self.assertEqual(len(zones), len(expected))
                    self.assertEqual([canvas.itemcget(z, 'fill') for z in zones], [e[1] for e in expected])
                    coordinates = [canvas.coords(z) for z in zones]
                    widths = [c[2] - c[0] for c in coordinates]
                    for segment_width in widths:
                        self.assertAlmostEqual(segment_width, widths[0])
                    for index, (_, _, a, b) in enumerate(expected):
                        for p in (a, a + (b - a) * 0.25, (a + b) / 2, b):
                            self.app.result['ai_prob'] = p
                            self.app.result['verdict'] = verdict_for(p, thresholds(self.app.model))
                            self.app._draw_probability()
                            x = canvas.coords(canvas.find_withtag('probability_line')[0])[0]
                            start, _, end, _ = coordinates[index]
                            self.assertGreaterEqual(x, start - 1e-8)
                            self.assertLessEqual(x, end + 1e-8)
                            self.assertAlmostEqual(x, start + (p - a) / (b - a) * (end - start))
        self.showerror.assert_not_called()
        self.assertEqual(self.errors, [])

    def assert_visible_layout(self, window, app):
        window.update()
        for button in app.buttons + [app.about_button]:
            self.assertTrue(button.winfo_ismapped(), button.cget('text'))
            self.assertGreaterEqual(button.winfo_rootx(), window.winfo_rootx())
            self.assertLessEqual(
                button.winfo_rootx() + button.winfo_width(), window.winfo_rootx() + window.winfo_width(),
            )
            self.assertLessEqual(
                button.winfo_rooty() + button.winfo_height(), window.winfo_rooty() + window.winfo_height(),
            )
            metrics = ruler_gui.font.Font(root=window, font=(app.family, 11))
            self.assertGreaterEqual(button.winfo_width(), metrics.measure(button.cget('text')) + app.px(14))
        self.assertLessEqual(app.input_prompt.winfo_reqwidth(), app.input_prompt.winfo_width())
        self.assertGreater(app.input_text.winfo_height(), app.px(100))

    def assert_probability_captions(self, app):
        canvas = app.prob_canvas
        captions = canvas.find_withtag('zone_caption')
        self.assertEqual([canvas.itemcget(item, 'text') for item in captions], ['更像人', '不确定', '很像 AI'])
        boxes = [canvas.bbox(item) for item in captions]
        for box in boxes:
            self.assertGreaterEqual(box[0], 0)
            self.assertLessEqual(box[2], canvas.winfo_width())
            self.assertLessEqual(box[3], canvas.winfo_height())
        for previous, current in zip(boxes, boxes[1:]):
            self.assertLess(previous[2], current[0])
        color = ruler_gui.VERDICT_COLORS.get(app.result['verdict'], ruler_gui.GREY)
        for tag in ('probability_triangle', 'probability_line'):
            self.assertEqual(canvas.itemcget(canvas.find_withtag(tag)[0], 'fill'), color)

    def test_real_dpi_minimum_layout(self):
        self.root.deiconify()
        self.root.geometry('{}x{}'.format(*self.root.minsize()))
        self.assert_visible_layout(self.root, self.app)
        self.assertAlmostEqual(self.app.scale, max(1.0, self.root.winfo_fpixels('1i') / 96.0))
        self.analyze_sample()
        self.assert_probability_captions(self.app)

    def test_scaled_minimum_layout_probability_and_panel_scroll(self):
        for scale in (1.0, 2.5):
            with self.subTest(scale=scale):
                # A separate Tcl interpreter avoids fonts cached at the native DPI.
                window = tk.Tk()
                window.tk.call('tk', 'scaling', scale * 96.0 / 72.0)
                window.report_callback_exception = lambda *error: self.errors.append(error)
                try:
                    with patch.object(window, 'winfo_fpixels', return_value=96.0 * scale):
                        app = ruler_gui.RulerApp(window, model=read_json(MODEL_PATH))
                    window.geometry('{}x{}'.format(*window.minsize()))
                    self.assert_visible_layout(window, app)
                    app.load_sample()
                    window.update()
                    app.analyze_text()
                    window.update()
                    self.assertEqual(app.prob_value.cget('text'), format_probability(app.result['ai_prob']))
                    self.assert_probability_captions(app)
                    app.model['thresholds'].update(human_low=0.01, ai_high=0.99)
                    app._draw_probability()
                    self.assert_probability_captions(app)
                    app.result['ai_prob'] = 0.0
                    for row, probability in zip(app.result['which'], (1.0, 0.0, 0.0)):
                        row['p'] = probability
                    app._render_result()
                    window.update()
                    self.assertEqual(app.prob_value.cget('text'), '1% 以下')
                    self.assertLessEqual(
                        app.prob_value.winfo_x() + app.prob_value.winfo_width(),
                        app.prob_value.master.winfo_width(),
                    )
                    for _, _, percent in app.which_rows.values():
                        self.assertGreaterEqual(percent.winfo_width(), app.small_font.measure(percent.cget('text')))
                        self.assertLessEqual(
                            percent.winfo_x() + percent.winfo_width(), app.which_frame.winfo_width(),
                        )
                    # Both inline and wrapped hints must fit the panel.
                    hint_rows = set()
                    for _, label, hint in app.marker_rows:
                        hint_rows.add(int(hint.grid_info()['row']))
                        self.assertLessEqual(hint.winfo_x() + hint.winfo_width(), label.master.winfo_width())
                    self.assertEqual(hint_rows, {0, 1})
                    app._replace_text('😀。\n' + ('中' * 149 + '。\n') * 20)
                    window.update()
                    app.analyze_text()
                    window.update()
                    chips = app.pieces_frame.winfo_children()
                    self.assertEqual(len(chips), len(app.result['pieces']))
                    self.assertGreater(len({chip.winfo_y() for chip in chips}), 1)
                    for chip in chips:
                        self.assertLessEqual(chip.winfo_x() + chip.winfo_width(), app.pieces_frame.winfo_width())
                        self.assertLessEqual(chip.winfo_y() + chip.winfo_height(), app.pieces_frame.winfo_height())
                        self.assertGreaterEqual(
                            chip.winfo_width(), app.small_font.measure(chip.cget('text')) + app.px(12),
                        )
                    canvas = app.right_canvas
                    self.assertLess(canvas.yview()[1], 1.0)
                    for event, options, direction in [
                        ('<MouseWheel>', {'delta': -120}, 1), ('<MouseWheel>', {'delta': 120}, -1),
                        ('<Button-5>', {}, 1), ('<Button-4>', {}, -1),
                    ]:
                        before = canvas.yview()[0]
                        app.marker_rows[0][1].event_generate(event, **options)
                        window.update()
                        self.assertGreater((canvas.yview()[0] - before) * direction, 0)
                    canvas.yview_moveto(0)
                    app.input_text.event_generate('<MouseWheel>', delta=-120)
                    window.update()
                    self.assertEqual(canvas.yview()[0], 0)
                    canvas.yview_moveto(1)
                    window.update()
                    self.assertAlmostEqual(canvas.yview()[1], 1.0)
                    bottom = canvas.winfo_rooty() + canvas.winfo_height()
                    for widget in (app.marker_rows[-1][1], app.evidence_titles['human']):
                        self.assertLess(widget.winfo_rooty(), bottom)
                    self.showerror.assert_not_called()
                    self.assertEqual(self.errors, [])
                finally:
                    window.destroy()


class WindowSizingTests(unittest.TestCase):
    def test_screen_caps_and_centring_at_96_and_240_dpi(self):
        self.assertEqual(
            ruler_gui.window_layout(2.5, 3000, 2000),
            ((2750, 1760), (2250, 1500), (125, 120)),
        )
        self.assertEqual(
            ruler_gui.window_layout(1.0, 1920, 1080),
            ((1100, 760), (900, 600), (410, 160)),
        )
        self.assertEqual(
            ruler_gui.window_layout(2.5, 1280, 720),
            ((1177, 633), (1177, 633), (51, 43)),
        )


if __name__ == '__main__':
    unittest.main()
