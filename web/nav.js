/* nav.js — one navigation menu for every model-graph page.
   Injects a grouped, searchable, keyboard-driven command menu (☰ / Cmd-K)
   into the page header, marks the current page, and rebuilds the marketing
   inline nav into Tools/Learn groups. Single source of truth: edit the links
   here, not seven page headers. Loaded via <script src="/nav.js" defer>. */
(function () {
  'use strict';
  var GH = 'https://github.com/QSchlegel/model-graph';
  var GROUPS = [
    {label: 'Tools', items: [
      {href: '/chat', name: 'chat', desc: 'chat with live per-layer internals'},
      {href: '/dashboard', name: 'dashboard', desc: 'block-level drill-down'},
      {href: '/agent', name: 'agent', desc: 'run a micro model as an agent'}]},
    {label: 'Learn', items: [
      {href: '/intro', name: 'intro', desc: 'transformers, along the papers'},
      {href: '/six-pager', name: 'six-pager', desc: 'the product narrative'},
      {href: '/blog/', name: 'blog', desc: 'notes & write-ups'}]},
    {label: 'Source', items: [
      {href: GH, name: 'GitHub', desc: 'the repository', ext: true}]}
  ];
  var HERE = (location.pathname.replace(/\/+$/, '') || '/');
  function current(href) {
    var h = href.replace(/\/+$/, '') || '/';
    if (h === '/blog') return HERE.indexOf('/blog') === 0;
    return h === HERE;
  }
  function mk(tag, props, kids) {
    var e = document.createElement(tag), k;
    if (props) for (k in props) {
      if (k === 'class') e.className = props[k];
      else if (k === 'text') e.textContent = props[k];
      else e.setAttribute(k, props[k]);
    }
    (kids || []).forEach(function (c) {
      if (c != null) e.appendChild(c.nodeType ? c : document.createTextNode(c));
    });
    return e;
  }

  /* ── styles (use each page's :root palette vars) ─────────────────────── */
  var css =
    '.mg-btn{font:12px var(--mono,monospace);color:var(--ink,#22344A);' +
    'background:rgba(255,255,255,.7);border:1px solid var(--ink,#22344A);' +
    'border-radius:6px;padding:5px 10px;cursor:pointer;white-space:nowrap;' +
    'display:inline-flex;gap:6px;align-items:center;line-height:1}' +
    '.mg-btn:hover{background:var(--ink,#22344A);color:var(--panel,#fff)}' +
    '.mg-btn .k{font-size:10px;opacity:.6;border:1px solid currentColor;' +
    'border-radius:3px;padding:0 4px}' +
    '.mg-ov{position:fixed;inset:0;z-index:9999;background:rgba(34,52,74,.34);' +
    '-webkit-backdrop-filter:blur(3px);backdrop-filter:blur(3px);' +
    'display:flex;align-items:flex-start;justify-content:center;' +
    'padding:12vh 16px 16px}' +
    '.mg-ov[hidden]{display:none}' +
    '.mg-panel{width:100%;max-width:460px;background:var(--panel,#fff);' +
    'border:1px solid var(--ink,#22344A);border-radius:12px;overflow:hidden;' +
    'box-shadow:0 18px 60px rgba(34,52,74,.34);font:13px var(--mono,monospace);' +
    'color:var(--ink,#22344A)}' +
    '.mg-q{width:100%;border:0;border-bottom:1px solid var(--line,#C9D2DB);' +
    'padding:13px 15px;font:14px var(--mono,monospace);color:var(--ink,#22344A);' +
    'background:transparent;outline:none}' +
    '.mg-list{max-height:56vh;overflow-y:auto;padding:6px}' +
    '.mg-glabel{font-size:10px;text-transform:uppercase;letter-spacing:.08em;' +
    'color:var(--muted,#6B7A8C);font-weight:600;padding:8px 10px 3px}' +
    '.mg-it{display:flex;gap:10px;align-items:baseline;padding:8px 10px;' +
    'border-radius:8px;cursor:pointer;text-decoration:none;color:inherit}' +
    '.mg-it .nm{font-weight:600;min-width:82px}' +
    '.mg-it .ds{color:var(--muted,#6B7A8C);font-size:11.5px}' +
    '.mg-it .cur{margin-left:auto;color:var(--heat2,#B02E3C);font-size:10px}' +
    '.mg-it.sel,.mg-it:hover{background:var(--paper,#EEF1F4)}' +
    '.mg-it.here{box-shadow:inset 3px 0 0 var(--heat1,#E39B2D)}' +
    '.mg-hint{border-top:1px solid var(--line,#C9D2DB);padding:7px 12px;' +
    'font-size:10.5px;color:var(--muted,#6B7A8C);display:flex;gap:12px}' +
    '.mg-hint b{color:var(--ink,#22344A);font-weight:600}' +
    '.mg-empty{padding:18px 12px;color:var(--muted,#6B7A8C);text-align:center}' +
    'header nav{overflow:visible;-webkit-mask-image:none;mask-image:none;' +
    'align-items:center}' +
    '@media(max-width:640px){.mg-ov{padding:8vh 10px}.mg-it .nm{min-width:70px}}';
  document.head.appendChild(mk('style', {text: css}));

  /* ── flat index for the palette / filtering ──────────────────────────── */
  var FLAT = [];
  GROUPS.forEach(function (g) {
    g.items.forEach(function (it) { FLAT.push({g: g.label, it: it}); });
  });

  /* ── command palette ─────────────────────────────────────────────────── */
  var ov = mk('div', {class: 'mg-ov', hidden: 'hidden', role: 'dialog',
    'aria-label': 'Navigation menu'});
  var q = mk('input', {class: 'mg-q', type: 'text',
    placeholder: 'Jump to a page…  (type to filter)', 'aria-label': 'Filter'});
  var list = mk('div', {class: 'mg-list'});
  var panel = mk('div', {class: 'mg-panel'}, [q, list,
    mk('div', {class: 'mg-hint'}, [
      mk('span', null, [mk('b', {text: '↑↓'}), ' move']),
      mk('span', null, [mk('b', {text: '↵'}), ' open']),
      mk('span', null, [mk('b', {text: 'esc'}), ' close'])])]);
  ov.appendChild(panel);
  document.body.appendChild(ov);

  var sel = 0, rows = [];
  function render(filter) {
    list.replaceChildren();
    rows = [];
    var f = (filter || '').toLowerCase().trim();
    var groups = {};
    FLAT.forEach(function (e) {
      var it = e.it;
      if (f && (it.name + ' ' + it.desc + ' ' + e.g).toLowerCase()
          .indexOf(f) < 0) return;
      (groups[e.g] = groups[e.g] || []).push(it);
    });
    var any = false;
    GROUPS.forEach(function (g) {
      var items = groups[g.label];
      if (!items || !items.length) return;
      any = true;
      list.appendChild(mk('div', {class: 'mg-glabel', text: g.label}));
      items.forEach(function (it) {
        var here = current(it.href);
        var row = mk('a', {class: 'mg-it' + (here ? ' here' : ''),
          href: it.href}, [
          mk('span', {class: 'nm', text: it.name}),
          mk('span', {class: 'ds', text: it.desc + (it.ext ? ' ↗' : '')}),
          here ? mk('span', {class: 'cur', text: 'current'}) : null]);
        if (it.ext) { row.target = '_blank'; row.rel = 'noopener'; }
        row.addEventListener('mousemove', function () { setSel(rows.indexOf(row)); });
        rows.push(row);
        list.appendChild(row);
      });
    });
    if (!any) list.appendChild(mk('div', {class: 'mg-empty',
      text: 'no match — press esc'}));
    sel = 0; paint();
  }
  function setSel(i) { if (i >= 0) { sel = i; paint(); } }
  function paint() {
    rows.forEach(function (r, i) { r.classList.toggle('sel', i === sel); });
    if (rows[sel]) rows[sel].scrollIntoView({block: 'nearest'});
  }
  function open() {
    ov.hidden = false; q.value = ''; render('');
    setTimeout(function () { q.focus(); }, 0);
  }
  function close() { ov.hidden = true; }
  function toggle() { ov.hidden ? open() : close(); }

  q.addEventListener('input', function () { render(q.value); });
  q.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel(Math.min(sel + 1, rows.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel(Math.max(sel - 1, 0)); }
    else if (e.key === 'Enter') { e.preventDefault(); if (rows[sel]) rows[sel].click(); }
    else if (e.key === 'Escape') { close(); }
  });
  ov.addEventListener('click', function (e) { if (e.target === ov) close(); });
  document.addEventListener('keydown', function (e) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || '')) ||
      e.target.isContentEditable;
    if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) {
      e.preventDefault(); toggle();
    } else if (e.key === '/' && !typing && ov.hidden) {
      e.preventDefault(); open();
    } else if (e.key === 'Escape' && !ov.hidden) { close(); }
  });

  function menuBtn(label) {
    var b = mk('button', {class: 'mg-btn', type: 'button',
      'aria-label': 'Open navigation menu'}, [label || '☰ menu',
      mk('span', {class: 'k', text: navigator.platform.indexOf('Mac') >= 0 ?
        '⌘K' : '^K'})]);
    b.addEventListener('click', toggle);
    return b;
  }

  /* ── mount: ONE menu button per header — the palette holds the links, so
        the header stays uncluttered and identical on every page ─────────── */
  function mount() {
    var header = document.querySelector('header');
    if (!header) { document.body.appendChild(mkFloat()); return; }
    var nav = header.querySelector('nav:not(#crumb)');
    var slot = nav || header.querySelector('.ctl') ||
      header.querySelector('.acts') || header;
    if (nav) nav.replaceChildren();     // drop the old inline link list
    var b = menuBtn('☰ menu');
    if (slot === header) b.style.marginLeft = 'auto';
    slot.appendChild(b);
  }
  function mkFloat() {
    var w = mk('div', null, [menuBtn('☰ menu')]);
    w.style.cssText = 'position:fixed;top:12px;right:12px;z-index:9998';
    return w;
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
