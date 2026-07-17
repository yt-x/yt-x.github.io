// 身份标识词轮播 —— 在此修改词语列表
var IDENTITY_WORDS = ['开源爱好者', '技术分享者', '终身学习者'];

// 动态加载组件
async function loadComponent(containerId, componentPath) {
    try {
        const response = await fetch(componentPath);
        if (!response.ok) {
            throw new Error(`Failed to load component: ${componentPath}`);
        }
        const html = await response.text();
        document.getElementById(containerId).innerHTML = html;
    } catch (error) {
        console.error('Error loading component:', error);
        document.getElementById(containerId).innerHTML =
            '<div class="text-center text-gray-500 py-8">组件加载失败</div>';
    }
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 加载音乐播放器组件
    loadComponent('music-container', 'components/music.html').then(() => {
        initMusicCard();
        // Trigger entrance reveal after the card is in DOM
        requestAnimationFrame(() => {
            const card = document.querySelector('.music-card');
            if (card) {
                card.classList.remove('pre-reveal');
                card.classList.add('reveal');
            }
        });
    });

    // 拉取最新博客文章
    loadLatestBlog();

    // 添加主题切换监听
    setupThemeToggle();

    // 身份标识词轮播
    startIdentityRotator();

    // 阅读进度条
    initProgressBar();
});

// 设置主题切换按钮
function setupThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const isDark = document.documentElement.classList.toggle('dark');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');

            // Rotation feedback
            themeToggle.classList.add('theme-spin');
            setTimeout(function () {
                themeToggle.classList.remove('theme-spin');
            }, 500);

            // 显示主题切换提示
            showToast(isDark ? '已切换到深色模式' : '已切换到浅色模式');
        });
    }
}

// 从 JSON / RSS 端点获取最新博客文章
async function fetchLatestPost() {
    // 优先 JSON（Hugo section output），失败回退到 RSS
    const endpoints = [
        'https://sgldbhxs.top/hugoblog/posts/index.json',
        'https://sgldbhxs.top/hugoblog/posts/index.xml'
    ];

    for (const url of endpoints) {
        try {
            const response = await fetch(url);
            if (!response.ok) continue;

            if (url.endsWith('.json')) {
                const data = await response.json();
                if (Array.isArray(data) && data.length > 0) {
                    return {
                        title: data[0].title || '无标题',
                        link: data[0].permalink || '#',
                        date: data[0].date || data[0].pubDate || '',
                        summary: data[0].summary || ''
                    };
                }
            } else {
                // RSS 回退
                const xmlText = await response.text();
                const parser = new DOMParser();
                const xmlDoc = parser.parseFromString(xmlText, 'text/xml');
                const items = xmlDoc.querySelectorAll('item');
                if (items.length > 0) {
                    const item = items[0];
                    return {
                        title: item.querySelector('title')?.textContent || '无标题',
                        link: item.querySelector('link')?.textContent || '#',
                        date: item.querySelector('pubDate')?.textContent || '',
                        summary: item.querySelector('description')?.textContent || ''
                    };
                }
            }
        } catch (e) {
            continue; // 尝试下一个端点
        }
    }
    return null;
}

// 格式化日期字符串
function formatDate(dateStr) {
    if (!dateStr) return '';
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr.slice(0, 10);
    return d.getFullYear() + '-' +
        String(d.getMonth() + 1).padStart(2, '0') + '-' +
        String(d.getDate()).padStart(2, '0');
}

// 拉取并渲染最新博客文章
async function loadLatestBlog() {
    const content = document.getElementById('blog-content');
    if (!content) return;

    const post = await fetchLatestPost();
    if (!post) {
        content.innerHTML = '<div class="text-center text-gray-400 py-6">博客内容加载失败</div>';
        return;
    }

    const formattedDate = formatDate(post.date);

    content.innerHTML = `
        <a href="${post.link}" target="_blank" rel="noopener noreferrer"
           class="reveal blog-item block -mx-3 px-3 py-2 transition-colors">
            <span class="museum-label mb-1 block">BLG. 01 / ${formattedDate || 'Recent'}</span>
            <h3 class="text-lg font-serif font-semibold text-ink dark:text-rice">
                ${post.title}
            </h3>
            ${post.summary ? `<p class="mt-2 text-ink-muted line-clamp-2 text-sm leading-relaxed">${post.summary}</p>` : ''}
            <span class="arrow-link inline-flex items-center gap-1 mt-3 text-xs text-ink-muted dark:text-rice/70 font-medium tracking-wide hover:text-ink dark:hover:text-rice transition-colors">
                阅读全文
                <svg class="icon text-xs" aria-hidden="true">
                    <use href="images/icons.svg#ri-arrow-right-line"></use>
                </svg>
            </span>
        </a>
    `;
}

// 显示提示信息
function showToast(message) {
    const toast = document.getElementById('toast');
    if (toast) {
        toast.textContent = message;
        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
        }, 2000);
    }
}

// 身份标识词轮播 —— 每 ~4s 交叉淡入淡出，上飘 4px
function startIdentityRotator() {
    const el = document.getElementById('identity-rotator');
    if (!el) return;

    // reduced-motion: 静态显示第一个词，不轮播
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let index = 0;
    let rotating = false;

    function rotate() {
        if (rotating) return;
        rotating = true;

        // 淡出 + 上飘
        el.style.opacity = '0';
        el.style.transform = 'translateY(-4px)';

        setTimeout(() => {
            index = (index + 1) % IDENTITY_WORDS.length;
            el.textContent = IDENTITY_WORDS[index];
            // 淡入
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
            rotating = false;
        }, 480);
    }

    setInterval(rotate, 4000);
}

// 阅读进度条 —— rAF 节流滚动更新
let _progressRAF = null;

function updateProgressBar() {
    if (_progressRAF) return;
    _progressRAF = requestAnimationFrame(() => {
        const scrollTop = window.scrollY || document.documentElement.scrollTop;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const progress = docHeight > 0 ? Math.min(scrollTop / docHeight, 1) : 1;
        const bar = document.getElementById('progress-bar');
        if (bar) bar.style.transform = 'scaleX(' + progress + ')';
        _progressRAF = null;
    });
}

function initProgressBar() {
    window.addEventListener('scroll', updateProgressBar, { passive: true });
    window.addEventListener('resize', updateProgressBar);
    updateProgressBar(); // 初始值
}
