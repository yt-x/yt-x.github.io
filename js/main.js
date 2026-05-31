// 动态加载组件
async function loadComponent(containerId, componentPath) {
    try {
        // 添加时间戳参数避免缓存
        const timestamp = new Date().getTime();
        const response = await fetch(`${componentPath}?v=${timestamp}`);
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
    // 初始化主题
    initTheme();
    
    // 加载个人资料组件
    loadComponent('profile-container', 'components/profile.html');
    
    // 加载技能组件
    loadComponent('skills-container', 'components/skills.html');
    
    // 加载项目卡片组件
    loadComponent('projects-container', 'components/projects.html');
    
    // 加载博客组件，完成后拉取最新文章
    loadComponent('blog-container', 'components/blog.html').then(() => {
        loadLatestBlog();
    });
    
    // 添加主题切换监听
    setupThemeToggle();
});

// 初始化主题
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
}

// 设置主题切换按钮
function setupThemeToggle() {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const isDark = document.documentElement.classList.toggle('dark');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            
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

    // 无独立卡片，内容直接流在琥珀色容器内
    content.innerHTML = `
        <a href="${post.link}" target="_blank"
           class="block -mx-3 px-3 py-2 rounded-lg hover:bg-amber-100/40 dark:hover:bg-gray-700/30 transition-colors">
            <h3 class="text-lg font-semibold text-gray-800 dark:text-white">
                ${post.title}
            </h3>
            ${formattedDate ? `<p class="text-sm text-gray-500 dark:text-gray-400 mt-1"><i class="ri-calendar-line mr-1"></i>${formattedDate}</p>` : ''}
            ${post.summary ? `<p class="mt-2 text-gray-600 dark:text-gray-300 line-clamp-2 text-sm leading-relaxed">${post.summary}</p>` : ''}
            <span class="inline-flex items-center gap-1 mt-3 text-sm text-amber-600 dark:text-amber-400 font-medium">
                阅读全文 <i class="ri-arrow-right-line text-xs"></i>
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
