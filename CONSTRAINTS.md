# CONSTRAINTS.md — 踩坑记录与项目约束

> 已知坑与硬约束的沉淀。每条：现象 → 根因 → 正确做法。
> 踩到新坑就追加（标注日期）；确认过时的条目标注 [已失效] 而非删除。

## 环境与本地预览

### C1. 必须起本地服务器，file:// 直接打开必败
- 现象：双击 index.html 后音乐卡等区域空白或报 CORS/fetch 错误。
- 根因：`js/main.js` 用 `fetch` 动态注入 `components/music.html`，file:// 协议下 fetch 被浏览器拦截。
- 做法：`npm start`（python -m http.server 8000）后访问 `http://localhost:8000`。

### C2. "Serving HTTP on :: port 8000" 但浏览器无响应（2026-07-17 踩）
- 现象：服务器显示已启动，浏览器一直无法连接。
- 根因：Windows 上 `SO_REUSEADDR` 语义与 Unix 不同——端口被旧的僵尸 python 进程占用时，新进程照样"启动成功"，但请求被路由到卡死的旧进程。
- 做法：`Get-NetTCPConnection -LocalPort 8000 -State Listen` 查占用 PID，`Stop-Process -Id <PID> -Force` 清掉后重启服务器。改端口（如 8123）可快速绕过。

### C3. 终端窗口点选文字会冻结服务器进程（2026-07-17 踩）
- 现象：终端里服务器"活着"但请求全部超时/无响应。
- 根因：Windows 控制台 QuickEdit 模式下，在窗口内单击/拖选进入标记状态，会暂停进程的输出乃至响应。
- 做法：在该终端窗口按 `Enter` 或 `Esc` 退出标记状态；或改用隐藏窗口启动（`Start-Process ... -WindowStyle Hidden`）。

## 浏览器缓存

### C4. 改了 JS 但页面行为还是旧的——先怀疑缓存，别怀疑代码（2026-07-17 踩）
- 现象：磁盘文件已更新、服务器返回的也是新字节，但页面执行的还是旧逻辑（本次：music-card.js 的 toggleMute "不存在"）。
- 根因：python http.server 不发 Cache-Control，Chrome 按启发式规则缓存 JS；长生命周期浏览器（含 Playwright MCP 的常驻浏览器）会持续命中旧缓存，`reload({ignoreCache:true})` 都未必可靠。
- 做法：排查顺序 = ① `Ctrl+F5` 硬刷新 → ② DevTools 勾 Disable cache → ③ CDP `Network.clearBrowserCache`。确认"服务器字节 vs 页面执行字节"是否一致再改代码。给用户交付时提醒硬刷新。

## Canvas / 动画

### C5. `IndexSizeError: The radius provided is negative`（2026-07-16 踩，已修）
- 现象：Canvas `arc()` 抛错，动画帧中断。
- 根因：交错动画元素 `startTime` 在未来（stagger 延迟），`elapsed = ts - startTime` 为负，负值经缓动公式算出负半径。
- 做法：绘制函数开头 guard `if (elapsed < 0) return;`。同类变体：延迟出生的元素 progress 为负导致位置错乱（`progress*progress` 会把负值变正）——凡是带 delay/stagger 的动画，先判 `elapsed < 0`。修复落点：`js/rain.js` drawRing 与 drops filter。

## 构建产物与图标

### C6. `css/tailwind.css` 与 `images/icons.svg` 是构建产物，禁止手改
- 做法：样式改 `css/input.css` / `styles.css` / `components.css` 后 `npm run build:css`；缺图标改 `scripts/build-icons.mjs` 的清单后 `npm run build:icons`，引用前先 grep `images/icons.svg` 确认 ID 存在（图标名写错不会报错，只是空白）。

## 项目契约（改动时的硬约束）

### C7. 主题机制：只认 `<html>.dark` 类
- CSS 写 `.dark` 变体；JS 读 `document.documentElement.classList.contains('dark')`（如 `js/rain.js` 生成时取色）。
- 禁止：JS 侧另起一套主题状态、`prefers-color-scheme` 直接驱动局部组件（入口 IIFE 已处理首屏）。

### C8. 新动画必须过双门槛
- ① `prefers-reduced-motion: reduce` 兜底（CSS 守卫块见两个 CSS 文件底部；JS 侧参考 `js/rain.js` 的 matchMedia 监听与 destroy）。
- ② 暗色模式适配（用调色板 token 的暗色变体，禁止写死只在浅色下可见的颜色）。

### C9. 音乐播放器 DOM ↔ JS 的 ID 契约
- `components/music.html` 的每个元素 ID 都被 `js/music-card.js` 的 `getMusicElement()` 引用。删/改 DOM 前必须审 JS：要么 JS 侧同步删除逻辑，要么保证可选链 null 安全。`.music-card`、`.pre-reveal` 两个类被 `js/main.js` 的入场触发依赖，不可改名。

### C10. index.html 是结构唯一事实源
- profile/技能/博客/项目已内联在 index.html；`components/` 下同名片段（profile.html 等）为历史遗留，改动以 index.html 为准。删除遗留文件前先 grep 全仓库确认无引用。

## 杂项

### C11. PowerShell 里 `git diff` 中文显示乱码
- 根因：控制台编码，不是文件损坏。
- 做法：核实中文内容用 Read 工具读文件，不要凭 diff 输出判断。

### C12. 0 字节死文件
- `js/theme.js`、`components/header.html` 为空文件，无引用价值；勿在它们基础上开发。

### C13. 既有 Console 报错（非缺陷，勿误判为回归）
- `favicon.ico` 404：站点无 favicon（待办见 TODO.md P3）。
- `https://sgldbhxs.top/hugoblog/posts/index.json` 404：外部博客 feed 端点当前不可用（待办见 TODO.md P1）。
- 验证标准：改动后 Console 不得出现**除上述两条外**的新报错。
