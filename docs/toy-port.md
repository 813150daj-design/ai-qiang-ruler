# 浏览器 / B站 Toy 移植版

把 Python 版的推理和文风规则移植到浏览器，无需 Python 服务或第三方推理 API。原模型权重按字节复制，输入文字在 Web Worker 中处理，不上传、不持久化。第一次测量需要下载约 8.7 MB 的模型。

原作者：[Claude大小姐在运营](https://space.bilibili.com/3747572619414174)。原视频：[《AI 腔是怎么来的？》](https://www.bilibili.com/video/BV1xNsV6CE7d/)。页面、许可说明、复制报告和封面均保留出处；原代码 MIT 许可与模型 CC BY-SA 4.0 许可不变。

## 本地运行

需要 Python 3.8+，不需要安装 Python 依赖。网页源码在 `web/`，上传产物生成到 `toy-dist/`：

```sh
python scripts/build_toy.py
python -m http.server 8897 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8897/toy-dist/index.html>。请通过 HTTP 访问；直接双击 HTML 不能正常加载模块、模型和 Worker。

保留的功能包括概率校准、三家文风倾向、长文分段、用词依据、十条文风检查、原文高亮和复制报告。不到 50 个汉字时不做模型判断，仍展示规则命中。文字修改后会提示重新测量，复制受浏览器限制时提供手动复制。

## 对照验证

另需 Node.js 18+，无需 npm 依赖：

```sh
python scripts/build_toy.py
python scripts/make_web_reference.py
node scripts/test_web.mjs
```

46 组对照覆盖原真实模型黄金样本，以及短文、规则、Emoji 和长文边界。比较 Python 与 JavaScript 的概率、三家倾向、用词依据、规则及标注位置，数值容差为 1e-9。浏览器使用 UTF-16 偏移，测试先把 Python 的 Unicode 码点偏移转换后再比较。

原 Python 代码和黄金样本未修改。本机 Python 3.10 上原有 42 项测试中有一项 `test_features.FeatureTests.test_golden` 的 `cjk_extension` 样例失败，涉及 U+4DBF 的骨架化预期差异；其余 41 项通过。

## Toy 打包

上传 `toy-dist/`，不要上传整个源码仓库。JS、CSS、JSON 的构建文件名带内容指纹，全部资源使用相对路径，适用于 Toy 子路径。源码使用 `.mjs`，构建为 `.js`，避免静态托管对 `.mjs` 返回非 JavaScript MIME 的问题。

`toy-assets/cover.png` 为 1200×900 封面，`toy-assets/icon.png` 为 512×512 图标。如需重新生成，本机安装 Pillow 后运行 `python scripts/make_toy_cover.py`；该辅助脚本使用 Windows 微软雅黑字体路径，可按操作系统替换。

当前提交地址为 <https://www.bilibili.com/toy/ai-qiang-rule/index.html>，仅链接可见，2026-11-15 已提交审核；能否访问以平台审核状态为准。

Toy 内容预检无错误或警告。托管的模型、规则、JS、Worker 和 CSS 已核验 HTTP 200、正确 MIME 与 SHA256 一致；本地浏览器已验证模型加载、示例分析、规则高亮和短文提示。页面已做响应式布局，桌面浏览器窄视口无横向溢出；手机真机和 B站 App 内 WebView 的输入、复制与模型加载仍待实测。
