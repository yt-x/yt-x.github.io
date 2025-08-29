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
    // 初始化主题
    initTheme();
    
    // 加载个人资料组件
    loadComponent('profile-container', 'components/profile.html');
    
    // 加载技能组件
    loadComponent('skills-container', 'components/skills.html');
    
    // 加载项目卡片组件
    loadComponent('projects-container', 'components/projects.html');
    
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