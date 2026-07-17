/**
 * Ambient raindrop effect — quiet museum rain
 * Thin streaks fall, spawn expanding ripple rings that fade.
 * Vanilla Canvas 2D, rAF loop, respects reduced-motion and visibility.
 */
(function () {
    'use strict';

    /* ---------- reduced-motion guard ---------- */
    var motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (motionQuery.matches) return;

    /* ---------- configuration ---------- */
    var CFG = {
        minInterval: 2500,       // ms between rain clusters (节奏保持不变)
        maxInterval: 5000,
        dropsMin: 2,             // 每阵雨的滴数
        dropsMax: 4,
        clusterSpread: 380,      // 同阵雨滴的时间散开范围 ms
        maxSimultaneous: 8,      // at most this many drops alive at once
        streakWidth: 2,          // px (logical, DPR-handled by transform)
        streakMinLen: 10,        // px
        streakMaxLen: 18,        // px
        streakDuration: 650,     // ms for streak to finish falling
        ringCount: 3,            // concentric ripples per landing
        ringStagger: 160,        // ms delay between rings
        ringMinRadius: 45,       // px max radius
        ringMaxRadius: 90,       // px max radius
        ringDuration: 2400,      // ms for ring to complete (水面上涟漪停留更久)
        ringLineWidth: 1.5,      // px
        ringStartOpacity: 0.55,  // at radius 0 (涟漪加深)
        splashRadius: 2.5,       // 入水瞬间的水花点
        splashDuration: 350,     // ms
        splashStartOpacity: 0.5,
        /* palette — read from theme at spawn time */
        dropLight: 'rgba(42,41,39,0.30)',
        dropDark: 'rgba(244,242,236,0.30)',
        ringLight: 'rgba(42,41,39,',
        ringDark: 'rgba(244,242,236,',
    };

    /* ---------- state ---------- */
    var canvas, ctx, dpr;
    var drops = [];
    var rings = [];
    var splashes = [];
    var animId = null;
    var paused = false;
    var lastDropTime = 0;
    var nextDropDelay = 0;
    var destroyed = false;

    /* ---------- helpers ---------- */
    function rand(min, max) {
        return min + Math.random() * (max - min);
    }

    function randInterval() {
        return rand(CFG.minInterval, CFG.maxInterval);
    }

    function isDark() {
        return document.documentElement.classList.contains('dark');
    }

    /* ---------- canvas setup ---------- */
    function createCanvas() {
        canvas = document.createElement('canvas');
        canvas.setAttribute('aria-hidden', 'true');
        canvas.style.cssText =
            'position:fixed;inset:0;z-index:0;pointer-events:none;display:block;';
        document.body.insertBefore(canvas, document.body.firstChild);
        ctx = canvas.getContext('2d');
    }

    function resize() {
        dpr = window.devicePixelRatio || 1;
        canvas.width = window.innerWidth * dpr;
        canvas.height = window.innerHeight * dpr;
        canvas.style.width = window.innerWidth + 'px';
        canvas.style.height = window.innerHeight + 'px';
    }

    /* ---------- spawn ---------- */
    function spawnDrop(ts, delay) {
        if (drops.length >= CFG.maxSimultaneous || destroyed) return;

        var x = rand(0, window.innerWidth);
        var targetY = rand(window.innerHeight * 0.05, window.innerHeight * 0.88);

        drops.push({
            x: x,
            targetY: targetY,
            streakLen: rand(CFG.streakMinLen, CFG.streakMaxLen),
            startTime: ts + (delay || 0),
            duration: CFG.streakDuration,
            color: isDark() ? CFG.dropDark : CFG.dropLight,
        });
    }

    /* 一阵雨：2~4 滴，时间上自然散开 */
    function spawnCluster(ts) {
        var count = Math.round(rand(CFG.dropsMin, CFG.dropsMax));
        for (var i = 0; i < count; i++) {
            spawnDrop(ts, rand(0, CFG.clusterSpread));
        }
    }

    function spawnSplash(x, y, ts) {
        splashes.push({
            x: x,
            y: y,
            startTime: ts,
            duration: CFG.splashDuration,
            colorBase: isDark() ? CFG.ringDark : CFG.ringLight,
        });
    }

    function spawnRings(x, y, ts) {
        var count = CFG.ringCount;
        var colorBase = isDark() ? CFG.ringDark : CFG.ringLight;

        for (var i = 0; i < count; i++) {
            rings.push({
                x: x,
                y: y,
                radius: 0,
                maxRadius: rand(CFG.ringMinRadius, CFG.ringMaxRadius),
                startTime: ts + i * CFG.ringStagger,
                duration: CFG.ringDuration,
                colorBase: colorBase,
            });
        }
    }

    /* ---------- draw ---------- */
    function drawDrop(drop, ts) {
        var elapsed = ts - drop.startTime;
        var progress = Math.min(elapsed / drop.duration, 1);
        /* quadratic ease-in: gravity acceleration */
        var eased = progress * progress;

        var startY = -drop.streakLen;
        var currentY = startY + (drop.targetY - startY) * eased;
        var topY = currentY - drop.streakLen;
        var botY = currentY;

        ctx.strokeStyle = drop.color;
        ctx.lineWidth = CFG.streakWidth;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(drop.x, topY);
        ctx.lineTo(drop.x, botY);
        ctx.stroke();
    }

    function drawRing(ring, ts) {
        var elapsed = ts - ring.startTime;
        if (elapsed < 0) return; // staggered ring not yet born
        var progress = Math.min(elapsed / ring.duration, 1);

        /* ease-out on expansion: fast start, slow end */
        var easedRadius = 1 - Math.pow(1 - progress, 2.5);
        var radius = easedRadius * ring.maxRadius;

        /* opacity: fade from startOpacity to 0 */
        var opacity = (1 - progress) * CFG.ringStartOpacity;

        ctx.strokeStyle = ring.colorBase + opacity.toFixed(3) + ')';
        ctx.lineWidth = CFG.ringLineWidth;
        ctx.beginPath();
        ctx.arc(ring.x, ring.y, radius, 0, Math.PI * 2);
        ctx.stroke();
    }

    function drawSplash(splash, ts) {
        var elapsed = ts - splash.startTime;
        if (elapsed < 0) return;
        var progress = Math.min(elapsed / splash.duration, 1);

        /* 水花点：迅速出现并原地淡去 */
        var opacity = (1 - progress) * CFG.splashStartOpacity;
        var radius = CFG.splashRadius * (1 + progress * 0.6);

        ctx.fillStyle = splash.colorBase + opacity.toFixed(3) + ')';
        ctx.beginPath();
        ctx.arc(splash.x, splash.y, radius, 0, Math.PI * 2);
        ctx.fill();
    }

    /* ---------- main loop ---------- */
    function tick(ts) {
        if (destroyed) return;

        if (paused) {
            animId = requestAnimationFrame(tick);
            return;
        }

        /* spawn check — 每次落下一阵雨 */
        if (ts - lastDropTime >= nextDropDelay) {
            lastDropTime = ts;
            nextDropDelay = randInterval();
            spawnCluster(ts);
        }

        /* clear — reset transform so clearRect uses physical pixels */
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        /* reapply DPR scaling */
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        /* process drops */
        drops = drops.filter(function (drop) {
            var elapsed = ts - drop.startTime;
            if (elapsed < 0) return true; // 簇内延迟，尚未出生
            if (elapsed >= drop.duration) {
                /* drop landed at targetY — 入水：水花点 + 涟漪 */
                spawnSplash(drop.x, drop.targetY, ts);
                spawnRings(drop.x, drop.targetY, ts);
                return false;
            }
            drawDrop(drop, ts);
            return true;
        });

        /* process splashes */
        splashes = splashes.filter(function (splash) {
            var elapsed = ts - splash.startTime;
            if (elapsed >= splash.duration) return false;
            drawSplash(splash, ts);
            return true;
        });

        /* process rings */
        rings = rings.filter(function (ring) {
            var elapsed = ts - ring.startTime;
            if (elapsed >= ring.duration) return false;
            drawRing(ring, ts);
            return true;
        });

        animId = requestAnimationFrame(tick);
    }

    /* ---------- lifecycle ---------- */
    function destroy() {
        destroyed = true;
        if (animId) {
            cancelAnimationFrame(animId);
            animId = null;
        }
        if (canvas && canvas.parentNode) {
            canvas.parentNode.removeChild(canvas);
        }
        drops.length = 0;
        rings.length = 0;
        splashes.length = 0;
    }

    function init() {
        createCanvas();
        resize();
        lastDropTime = performance.now();
        nextDropDelay = randInterval();

        window.addEventListener('resize', resize);
        document.addEventListener('visibilitychange', function () {
            paused = document.hidden;
        });

        animId = requestAnimationFrame(tick);
    }

    /* react to reduced-motion preference changing mid-session */
    motionQuery.addEventListener('change', function (e) {
        if (e.matches) {
            destroy();
        }
        /* re-enabling after reduce→no-preference is not supported
           (requires full re-init; rare in practice) */
    });

    /* start when DOM is ready */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
