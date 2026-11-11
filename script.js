(() => {
  'use strict';
  const menu = document.querySelector('.menu-toggle');
  const nav = document.querySelector('#site-nav');
  function closeMenu() {
    menu.setAttribute('aria-expanded', 'false');
    menu.setAttribute('aria-label', 'Open navigation');
    nav.classList.remove('is-open');
  }
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    nav.classList.toggle('is-open', open);
  });
  nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });

  const tabs = [...document.querySelectorAll('[role="tab"]')];
  function activateTab(tab, focus = false) {
    tabs.forEach(item => {
      const active = item === tab;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(item.getAttribute('aria-controls'));
      panel.hidden = !active;
      if (!active) panel.querySelectorAll('video').forEach(video => video.pause());
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activateTab(tab));
    tab.addEventListener('keydown', event => {
      const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      activateTab(tabs[next], true);
    });
  });

  document.querySelectorAll('[data-scholar-placeholder]').forEach(link => {
    if (link.href.includes('SCHOLAR_ID_')) {
      link.setAttribute('aria-disabled', 'true');
      link.addEventListener('click', event => event.preventDefault());
    } else {
      link.removeAttribute('aria-disabled');
      link.title = 'Google Scholar';
    }
  });

  const videoConfig = window.NWM_VIDEOS || {};
  document.querySelectorAll('[data-video-slot]').forEach(slot => {
    const source = videoConfig[slot.dataset.videoSlot];
    if (typeof source !== 'string' || !source.trim()) return;
    const video = slot.querySelector('video');
    const placeholder = slot.querySelector('.video-placeholder');
    video.src = source.trim();
    video.hidden = false;
    placeholder.hidden = true;
    video.addEventListener('error', () => {
      video.hidden = true;
      placeholder.hidden = false;
      placeholder.querySelector('span:last-child').textContent = 'Video unavailable';
    });
  });

  const gallery = document.querySelector('.generation-gallery');
  const galleryVideos = [...document.querySelectorAll('.generation-video')];
  const galleryToggle = document.getElementById('gallery-toggle');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let galleryEnabled = !reducedMotion.matches;
  let galleryVisible = false;
  function updateGalleryButton() {
    const playing = galleryVideos.some(video => !video.paused && !video.ended);
    galleryToggle.textContent = playing ? 'Pause videos' : 'Play videos';
    galleryToggle.setAttribute('aria-pressed', String(playing));
  }
  function pauseGallery() {
    galleryVideos.forEach(video => video.pause());
    updateGalleryButton();
  }
  function playGallery() {
    if (document.hidden) return;
    galleryVideos.forEach(video => { video.play().catch(updateGalleryButton); });
  }
  galleryVideos.forEach(video => {
    video.addEventListener('play', updateGalleryButton);
    video.addEventListener('pause', updateGalleryButton);
  });
  galleryToggle.addEventListener('click', () => {
    if (galleryVideos.some(video => !video.paused)) {
      galleryEnabled = false;
      pauseGallery();
    } else {
      galleryEnabled = true;
      playGallery();
    }
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      galleryVisible = entries[0].isIntersecting;
      if (galleryVisible && galleryEnabled) playGallery();
      else pauseGallery();
    }, { threshold: 0.15 }).observe(gallery);
  } else {
    galleryVisible = true;
    if (galleryEnabled) playGallery();
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      pauseGallery();
      document.querySelectorAll('[data-video-slot] video').forEach(video => video.pause());
    } else if (galleryEnabled && galleryVisible) playGallery();
  });
  reducedMotion.addEventListener('change', event => {
    if (event.matches) { galleryEnabled = false; pauseGallery(); }
  });

  const dialog = document.getElementById('figure-dialog');
  const dialogImage = document.getElementById('dialog-image');
  let figureTrigger = null;
  let previousOverflow = '';
  document.querySelectorAll('[data-zoom]').forEach(link => link.addEventListener('click', event => {
    if (typeof dialog.showModal !== 'function') return;
    event.preventDefault();
    figureTrigger = link;
    dialogImage.src = link.href;
    dialogImage.alt = link.querySelector('img').alt;
    document.getElementById('dialog-title').textContent = link.closest('figure').querySelector('figcaption').textContent.trim();
    previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.showModal();
  }));
  document.getElementById('close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => {
    document.body.style.overflow = previousOverflow;
    if (figureTrigger) figureTrigger.focus();
  });

  const copyButton = document.getElementById('copy-citation');
  const copyStatus = document.getElementById('copy-status');
  copyButton.addEventListener('click', async () => {
    const citation = document.getElementById('bibtex').textContent;
    try {
      if (!navigator.clipboard || !window.isSecureContext) throw new Error('Clipboard API unavailable');
      await navigator.clipboard.writeText(citation);
      copyStatus.textContent = 'BibTeX copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.getElementById('bibtex'));
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = 'Citation selected. Press Ctrl+C (or Cmd+C) to copy.';
    }
  });

  if ('IntersectionObserver' in window) {
    const navLinks = [...nav.querySelectorAll('a')];
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (!visible.length) return;
      navLinks.forEach(link => {
        if (link.hash === `#${visible[0].target.id}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }, { rootMargin: '-15% 0px -55% 0px', threshold: 0 });
    navLinks.forEach(link => { const section = document.querySelector(link.hash); if (section) observer.observe(section); });
  }
})();
