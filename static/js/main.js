/* ═══════════════════════════════════════════
   Deep Shield Mail — Global JS Utilities
   ═══════════════════════════════════════════ */

/* ── Inject animation keyframes into <head> ── */
(function injectAnimationStyles() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes logoFloatSm {
      0%   { transform: translateY(0px) scale(1);    filter: drop-shadow(0 0 6px rgba(37,99,235,0.7)); }
      50%  { transform: translateY(-4px) scale(1.07); filter: drop-shadow(0 0 14px rgba(56,189,248,1)); }
      100% { transform: translateY(0px) scale(1);    filter: drop-shadow(0 0 6px rgba(37,99,235,0.7)); }
    }
    @keyframes logoFloatHero {
      0%   { transform: translateY(0px) scale(1);     filter: drop-shadow(0 0 30px rgba(37,99,235,0.6)); }
      50%  { transform: translateY(-14px) scale(1.06); filter: drop-shadow(0 0 60px rgba(56,189,248,0.95)); }
      100% { transform: translateY(0px) scale(1);     filter: drop-shadow(0 0 30px rgba(37,99,235,0.6)); }
    }
    @keyframes ringPulse {
      0%, 100% { transform: scale(1);    opacity: 0.5; }
      50%       { transform: scale(1.05); opacity: 1;   }
    }
    @keyframes floatIcon {
      0%, 100% { transform: translateY(0px); }
      50%       { transform: translateY(-8px); }
    }
    @keyframes glowPulse {
      0%, 100% { box-shadow: 0 0 10px rgba(37,99,235,0.4); }
      50%       { box-shadow: 0 0 28px rgba(56,189,248,0.9); }
    }

    /* Navbar logo wrapper */
    .nav-logo-wrap {
      animation: logoFloatSm 4.5s ease-in-out infinite;
      display: inline-flex;
      transform-origin: center center;
    }

    /* Hero shield wrapper */
    .shield-logo-hero {
      animation: logoFloatHero 6s ease-in-out infinite;
      display: inline-flex;
      transform-origin: center center;
      width: 300px;
      height: 300px;
    }

    /* Sidebar brand logo */
    .sidebar-brand .nav-logo-wrap {
      animation: logoFloatSm 5s ease-in-out infinite;
    }

    /* Gmail login page hero logo */
    .gmail-login-logo {
      animation: logoFloatHero 6s ease-in-out infinite;
      display: inline-flex;
      transform-origin: center center;
    }
  `;
  document.head.appendChild(style);
})();

/* ── Particles ── */
function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W = canvas.width = window.innerWidth;
  let H = canvas.height = window.innerHeight;

  const particles = Array.from({ length: 55 }, () => ({
    x: Math.random() * W, y: Math.random() * H,
    r: Math.random() * 1.5 + 0.3,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    alpha: Math.random() * 0.4 + 0.1,
    color: Math.random() > 0.6 ? '#38BDF8' : '#2563EB',
  }));

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + Math.round(p.alpha * 255).toString(16).padStart(2, '0');
      ctx.fill();
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
    });
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 100) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(37,99,235,${0.08 * (1 - dist / 100)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
  window.addEventListener('resize', () => {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  });
}

/* ── Flash messages auto-dismiss ── */
function initFlash() {
  const msgs = document.querySelectorAll('.flash-msg');
  msgs.forEach(msg => {
    setTimeout(() => {
      msg.style.opacity = '0';
      msg.style.transform = 'translateX(20px)';
      msg.style.transition = 'all 0.3s ease';
      setTimeout(() => msg.remove(), 300);
    }, 4000);
  });
}

/* ── Number counter animation ── */
function animateCount(el, target) {
  const duration = 1200;
  const start = performance.now();
  function update(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(ease * target);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

/* ── Shield Logo SVG ── */
function getShieldLogoSVG(size) {
  const uid = size + Math.random().toString(36).slice(2, 6);
  return `<svg width="${size}" height="${size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="sg${uid}" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#2563EB"/>
        <stop offset="100%" stop-color="#38BDF8"/>
      </linearGradient>
      <linearGradient id="sg2${uid}" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.3"/>
        <stop offset="100%" stop-color="#2563EB" stop-opacity="0"/>
      </linearGradient>
      <filter id="gf${uid}" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="2.5" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <!-- Outer glow ring -->
    <ellipse cx="24" cy="46" rx="16" ry="3" fill="url(#sg2${uid})" opacity="0.6"/>
    <!-- Shield body -->
    <path d="M24 3L7 9.5V22C7 32.5 14.5 42.5 24 45.5C33.5 42.5 41 32.5 41 22V9.5L24 3Z"
      fill="url(#sg${uid})" filter="url(#gf${uid})" opacity="0.92"/>
    <!-- Shield inner highlight -->
    <path d="M24 3L7 9.5V22C7 32.5 14.5 42.5 24 45.5C33.5 42.5 41 32.5 41 22V9.5L24 3Z"
      fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="0.8"/>
    <!-- Envelope body -->
    <rect x="13" y="17" width="22" height="15" rx="2" fill="white" opacity="0.96"/>
    <!-- Envelope flap line -->
    <path d="M13 19.5L24 27L35 19.5" stroke="#2563EB" stroke-width="1.6"
      fill="none" stroke-linecap="round"/>
  
  </svg>`;
}

/* ── Gmail SVG ── */
function getGmailSVG(size) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
    <rect width="48" height="48" rx="6" fill="white"/>
    <path d="M6 14h36v22a2 2 0 01-2 2H8a2 2 0 01-2-2V14z" fill="#f8f9fa"/>
    <path d="M6 14l18 16L42 14" fill="none" stroke="#EA4335" stroke-width="2.5" stroke-linejoin="round"/>
    <path d="M6 36l14-13" stroke="#34A853" stroke-width="2" fill="none"/>
    <path d="M42 36L28 23" stroke="#4285F4" stroke-width="2" fill="none"/>
    <path d="M6 14h36" stroke="#DADCE0" stroke-width="0.5" fill="none"/>
  </svg>`;
}

/* ── Inject logos ── */
function injectLogos() {
  /* Navbar small logos */
  document.querySelectorAll('[data-logo="shield-sm"]').forEach(el => {
    el.innerHTML = getShieldLogoSVG(36);
  });

  /* Hero big logo */
  document.querySelectorAll('[data-logo="shield-hero"]').forEach(el => {
    el.innerHTML = getShieldLogoSVG(300);
    el.style.display = 'inline-flex';
    el.style.animation = 'logoFloatHero 6s ease-in-out infinite';
    el.style.transformOrigin = 'center center';
    el.style.filter = 'drop-shadow(0 0 40px rgba(37,99,235,0.7))';
  });

  /* Gmail logos */
  document.querySelectorAll('[data-logo="gmail"]').forEach(el => {
    el.innerHTML = getGmailSVG(parseInt(el.dataset.size) || 24);
  });
}

/* ── Init all on DOM ready ── */
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initFlash();
  injectLogos();

  document.querySelectorAll('[data-count]').forEach(el => {
    animateCount(el, parseInt(el.dataset.count));
  });
});
