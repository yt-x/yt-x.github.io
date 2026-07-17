# AGENTS.md — HomePage 项目指南

> AI 协作者入口。任务开始前按顺序读：本文件 → `TODO.md`（当前任务）→ `CONSTRAINTS.md`（坑与约束）。
> 完成任务后：更新 `TODO.md`；踩到新坑：追加 `CONSTRAINTS.md`。

## 概览

个人主页静态站点（SGLDBHXS），GitHub Pages 部署（仓库根目录发布，`.github/workflows/deploy.yml`，域名见 `CNAME`）。
设计语言：水墨/博物馆风——Noto Serif SC + Inter，纸墨双色 + 粉色 accent，克制动效，"开馆→看展(EXHIBIT)→页码(Colophon)"概念体系。

## 技术栈

- 原生 HTML/CSS/JS，**无前端框架、无运行时依赖**
- Tailwind CSS 3.4 仅作构建层：`css/input.css` → `css/tailwind.css`
- 图标：remixicon → `scripts/build-icons.mjs` → `images/icons.svg`（SVG sprite，`<use href="...#ri-xxx">`引用）

## 目录职责

```text
index.html          # 单页全部结构（profile/技能/博客/项目/Colophon 已内联）
components/         # 仅 music.html 被动态加载；其余为历史遗留片段，以 index.html 为准
css/styles.css      # 调色板变量、museum-label、动效系统、reduced-motion
css/components.css  # 组件样式：音乐条、项目卡、均衡器等
css/input.css       # Tailwind 源（3 行指令）
css/tailwind.css    # 构建产物，禁止手改
js/main.js          # 组件加载、主题切换、博客渲染、身份词轮播、阅读进度条
js/music-card.js    # 音乐播放器全部逻辑（约 490 行，顶层函数，无模块封装）
js/rain.js          # 雨滴涟漪 Canvas 背景（IIFE，z-0，内容层 z-10）
js/theme.js         # 0 字节死文件
data/               # playlist.json（曲库）、now-playing.json（Actions 定时生成的网易云记录）
scripts/            # build-icons.mjs、fetch-netease-music.js
.github/workflows/  # deploy.yml（Pages）、fetch-music.yml（每 6h 更新 now-playing.json）
```

## 核心机制（改代码前必读）

1. **组件注入**：`js/main.js` 的 `loadComponent('music-container', 'components/music.html')` 注入后调 `initMusicCard()`，再用 rAF 移除 `.pre-reveal` 加 `.reveal` 触发入场。**必须起本地服务器预览**，file:// 下 fetch 失败。
2. **主题**：`<html>.dark` 类开关 + localStorage `theme`。CSS 写 `.dark` 变体；JS 读 `document.documentElement.classList`。禁止 JS 侧另起主题判断。
3. **动效体系**：`css/styles.css` 自定义属性 `--ease-museum/--ease-out-smooth/--dur-*`；入场用 `.reveal` + `.reveal-delay-1..6`（`animation-fill-mode: backwards`）。**任何新动画必须配 `prefers-reduced-motion` 兜底**（两个 CSS 文件底部已有守卫块）。
4. **音乐播放器 ID 契约**：`components/music.html` 的元素 ID 与 `js/music-card.js` 的 `getMusicElement()` 一一对应。删/改 DOM 先审 JS 引用。`.is-playing` 类（JS 在 #music-player 上切换）驱动均衡器显隐。
5. **调色板**：`--ink #2a2927 / --rice #f4f2ec / --stone / --accent #db2777`，暗色 `--ink-dark/--ink-card/--ink-line`。新样式只用这些 token，不引入新色。
6. **页面层级**：雨滴 Canvas `z-0` < 内容包裹层 `z-10` < 阅读进度条 `z-50`（均 `pointer-events: none`）。

## 命令

```powershell
npm run dev          # build:icons + build:css + watch
npm run build        # 一次性构建（icons + css）
npm start            # python -m http.server 8000 → http://localhost:8000
node --check js/<file>.js   # JS 语法检查（改 JS 后必跑）
```

## 禁忌（违反即返工）

- 禁止手改 `css/tailwind.css`、`images/icons.svg`（构建产物）。缺图标：改 `scripts/build-icons.mjs` 的清单后 `npm run build:icons`，引用前先 grep 确认 ID 存在。
- 禁止引入任何 npm 依赖/前端框架/CSS 动画库。
- 禁止改动配色、字体、间距体系；动效不得花哨（无弹性缓动、无粒子、无渐变动画）。
- 新功能默认双验证：暗色模式 + reduced-motion。
- Git 只读操作除非用户明确要求提交。

## 协作工作流

1. 读 `TODO.md` 确认当前待办与验收标准。
2. 改动后按 `CONSTRAINTS.md` 自查是否踩已知坑（尤其浏览器缓存与本地服务器两条）。
3. 验证：`node --check` + `npm run build` + 浏览器实测（页面无新增 Console 报错；既有报错只有 favicon 404 与博客 feed 404）。
4. 收尾：更新 `TODO.md`（移入已完成+日期+落点文件）；新坑入 `CONSTRAINTS.md`。
