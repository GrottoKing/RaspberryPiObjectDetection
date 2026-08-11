/* Camera sightings log -- client.
 *
 * Polls /api/state, keeps a local map of sightings keyed by id, and renders
 * them grouped by category. The interesting bit is the hover preview: the
 * snapshot follows the cursor and only disappears once the pointer has moved a
 * decent distance away from the row it belongs to.
 */
(function () {
  "use strict";

  var POLL_MS = parseInt(document.body.dataset.pollMs, 10) || 1500;

  // How far (px) the cursor must get from the hovered row before the preview
  // fades out. Generous, so drifting between nearby rows feels continuous.
  var HIDE_DISTANCE = 170;
  var CURSOR_OFFSET = 20;

  var els = {
    log: document.getElementById("log"),
    categories: document.getElementById("categoryList"),
    labels: document.getElementById("labelList"),
    search: document.getElementById("search"),
    groupToggle: document.getElementById("groupToggle"),
    status: document.getElementById("statusLine"),
    pillLive: document.getElementById("pillLive"),
    pillTotal: document.getElementById("pillTotal"),
    pillFps: document.getElementById("pillFps"),
    empty: document.getElementById("emptyState"),
    loadMore: document.getElementById("loadMore"),
    preview: document.getElementById("preview"),
    previewImg: document.getElementById("previewImg"),
    previewCaption: document.getElementById("previewCaption"),
    lightbox: document.getElementById("lightbox"),
    lightboxImg: document.getElementById("lightboxImg"),
    lightboxCaption: document.getElementById("lightboxCaption"),
    streamWrap: document.getElementById("streamWrap"),
    streamImg: document.getElementById("streamImg"),
    toggleStream: document.getElementById("toggleStream"),
    togglePause: document.getElementById("togglePause"),
    clearLog: document.getElementById("clearLog")
  };

  var state = {
    entries: new Map(),      // id -> entry
    liveIds: new Set(),      // ids currently in view
    category: "",
    query: "",
    maxId: 0,
    sinceTs: 0,
    oldestId: null,
    paused: false,
    knownCategories: new Set(),
    firstLoad: true
  };

  /* ------------------------------------------------------------ utilities */

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function clockTime(epochSeconds) {
    var d = new Date(epochSeconds * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function agoText(epochSeconds) {
    var seconds = Math.max(0, Date.now() / 1000 - epochSeconds);
    if (seconds < 45) return "just now";
    if (seconds < 3600) return Math.round(seconds / 60) + "m ago";
    if (seconds < 86400) return Math.round(seconds / 3600) + "h ago";
    return Math.round(seconds / 86400) + "d ago";
  }

  function durationText(seconds) {
    if (seconds < 1) return "under a second";
    if (seconds < 60) return Math.round(seconds) + "s";
    return Math.floor(seconds / 60) + "m " + Math.round(seconds % 60) + "s";
  }

  /* -------------------------------------------------------------- polling */

  function poll() {
    if (state.paused) return;
    var url = "/api/state?since_id=" + state.maxId + "&since_ts=" + state.sinceTs;
    fetch(url, { cache: "no-store" })
      .then(function (response) { return response.json(); })
      .then(applyState)
      .catch(function (err) {
        els.status.textContent = "lost contact with the Pi - retrying";
        console.warn("poll failed", err);
      });
  }

  function applyState(data) {
    var changed = false;
    var freshIds = [];

    (data.entries || []).forEach(function (entry) {
      if (!state.entries.has(entry.id) && !state.firstLoad) freshIds.push(entry.id);
      state.entries.set(entry.id, entry);
      if (entry.id > state.maxId) state.maxId = entry.id;
      if (state.oldestId === null || entry.id < state.oldestId) state.oldestId = entry.id;
      changed = true;
    });

    state.sinceTs = data.server_time || state.sinceTs;

    state.liveIds = new Set((data.live || [])
      .map(function (item) { return item.id; })
      .filter(function (id) { return id != null; }));

    renderCategories(data.categories || []);
    renderLabels(data.labels || []);
    renderStatus(data);

    if (changed || state.firstLoad) render(freshIds);
    state.firstLoad = false;
  }

  function renderStatus(data) {
    var status = data.status || {};
    var stats = data.stats || {};
    var bits = [];
    if (status.camera) bits.push("camera: " + status.camera);
    if (status.backend) bits.push("detector: " + (status.backend_detail || status.backend));
    if (status.error) bits.push("error: " + status.error);
    els.status.textContent = bits.join("  ·  ") || "running";

    var liveCount = (data.live || []).length;
    els.pillLive.innerHTML = "<b>" + liveCount + "</b> in view";
    els.pillLive.classList.toggle("alive", liveCount > 0);
    els.pillTotal.innerHTML = "<b>" + (stats.total || 0) + "</b> logged";
    els.pillFps.innerHTML = "<b>" + (status.fps != null ? status.fps : 0) + "</b> fps";
  }

  /* ------------------------------------------------------------ rendering */

  function renderCategories(categories) {
    var total = categories.reduce(function (sum, c) { return sum + c.count; }, 0);
    var html = '<button class="cat' + (state.category === "" ? " active" : "") +
      '" data-category="" type="button"><span class="cat-name">Everything</span>' +
      '<span class="cat-count">' + total + "</span></button>";

    categories.forEach(function (item) {
      // Flash categories the first time they ever show up -- the log
      // literally grows new sections as the camera meets new things.
      var isNew = !state.knownCategories.has(item.name) && !state.firstLoad;
      state.knownCategories.add(item.name);
      html += '<button class="cat' + (state.category === item.name ? " active" : "") +
        (isNew ? " fresh" : "") + '" data-category="' + esc(item.name) +
        '" type="button"><span class="cat-name">' + esc(item.name) +
        '</span><span class="cat-count">' + item.count + "</span></button>";
    });
    els.categories.innerHTML = html;
  }

  function renderLabels(labels) {
    els.labels.innerHTML = labels.slice(0, 12).map(function (item) {
      return "<li><span>" + esc(item.label) + "</span><b>" + item.count + "</b></li>";
    }).join("");
  }

  function matches(entry) {
    if (state.category && entry.category !== state.category) return false;
    if (!state.query) return true;
    var haystack = (entry.description + " " + entry.label + " " + entry.category +
      " " + Object.values(entry.attributes || {}).join(" ")).toLowerCase();
    return haystack.indexOf(state.query) !== -1;
  }

  function rowHtml(entry, isFresh) {
    var live = state.liveIds.has(entry.id);
    var chips = [];
    if (live) chips.push('<span class="chip live">in view now</span>');
    chips.push('<span class="chip">' + esc(entry.label) + "</span>");
    Object.keys(entry.attributes || {}).forEach(function (key) {
      chips.push('<span class="chip">' + esc(key) + ": " + esc(entry.attributes[key]) + "</span>");
    });
    if (entry.duration >= 1) {
      chips.push('<span class="chip">seen for ' + durationText(entry.duration) + "</span>");
    }

    var swatch = entry.swatch ? ' style="background:' + esc(entry.swatch) + '"' : "";
    var snapshotAttr = entry.snapshot ? ' data-snapshot="' + esc(entry.snapshot) + '"' : "";

    return '<div class="row' + (isFresh ? " new" : "") + '"' + snapshotAttr +
      ' data-id="' + entry.id + '" tabindex="0"' +
      ' data-caption="' + esc(entry.description + " — " + clockTime(entry.first_seen)) + '">' +
      '<span class="swatch"' + swatch + "></span>" +
      '<div class="desc"><div class="title">' + esc(entry.description) + "</div>" +
      '<div class="meta">' + chips.join("") + "</div></div>" +
      '<div class="when">' + clockTime(entry.first_seen) +
      '<span class="conf">' + agoText(entry.last_seen) + " · " +
      Math.round(entry.confidence * 100) + "%</span></div></div>";
  }

  function render(freshIds) {
    var fresh = new Set(freshIds || []);
    var visible = Array.from(state.entries.values())
      .filter(matches)
      .sort(function (a, b) { return b.id - a.id; });

    els.empty.hidden = visible.length > 0;
    els.loadMore.hidden = state.entries.size === 0;

    var html;
    if (els.groupToggle.checked) {
      var groups = new Map();
      visible.forEach(function (entry) {
        if (!groups.has(entry.category)) groups.set(entry.category, []);
        groups.get(entry.category).push(entry);
      });
      // Newest activity first, so whatever just happened is at the top.
      var ordered = Array.from(groups.entries()).sort(function (a, b) {
        return b[1][0].id - a[1][0].id;
      });
      html = ordered.map(function (pair) {
        return '<section class="group"><div class="group-head"><h2>' +
          esc(pair[0]) + "</h2><span>" + pair[1].length + " sighting" +
          (pair[1].length === 1 ? "" : "s") + '</span></div><div class="rows">' +
          pair[1].map(function (e) { return rowHtml(e, fresh.has(e.id)); }).join("") +
          "</div></section>";
      }).join("");
    } else {
      html = '<div class="rows">' +
        visible.map(function (e) { return rowHtml(e, fresh.has(e.id)); }).join("") +
        "</div>";
    }

    // Preserve the hover preview across re-renders: remember which row was
    // active and re-attach to its replacement if it is still on the page.
    var activeId = hover.rowId;
    els.log.innerHTML = html;
    if (activeId != null) {
      var replacement = els.log.querySelector('.row[data-id="' + activeId + '"]');
      if (replacement) hover.row = replacement;
      else hidePreview();
    }
  }

  /* -------------------------------------------------- cursor-follow preview */

  var hover = { row: null, rowId: null, x: 0, y: 0 };

  function distanceToRect(x, y, rect) {
    var dx = Math.max(rect.left - x, 0, x - rect.right);
    var dy = Math.max(rect.top - y, 0, y - rect.bottom);
    return Math.hypot(dx, dy);
  }

  function positionPreview(x, y) {
    var box = els.preview.getBoundingClientRect();
    var width = box.width || 332;
    var height = box.height || 220;

    // Prefer down-right of the cursor; flip when we would run off screen.
    var left = x + CURSOR_OFFSET;
    var top = y + CURSOR_OFFSET;
    if (left + width > window.innerWidth - 8) left = x - width - CURSOR_OFFSET;
    if (top + height > window.innerHeight - 8) top = y - height - CURSOR_OFFSET;
    // Flipping can still overshoot on a small window, so clamp to the viewport.
    left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
    top = Math.max(8, Math.min(top, window.innerHeight - height - 8));

    els.preview.style.transform =
      "translate3d(" + Math.round(left) + "px," + Math.round(top) + "px,0) scale(1)";
  }

  function showPreview(row, x, y) {
    var snapshot = row.getAttribute("data-snapshot");
    if (!snapshot) return;
    hover.row = row;
    hover.rowId = row.getAttribute("data-id");
    hover.x = x;
    hover.y = y;
    if (els.previewImg.getAttribute("src") !== snapshot) {
      els.previewImg.setAttribute("src", snapshot);
    }
    els.previewCaption.textContent = row.getAttribute("data-caption") || "";
    positionPreview(x, y);
    els.preview.classList.add("visible");
  }

  // Until the image has loaded, the preview's measured height is whatever the
  // previous snapshot was -- so a portrait image would be placed as if it were
  // landscape and hang off the bottom. Reposition once the real size is known.
  els.previewImg.addEventListener("load", function () {
    if (hover.row) positionPreview(hover.x, hover.y);
  });

  function hidePreview() {
    hover.row = null;
    hover.rowId = null;
    els.preview.classList.remove("visible");
  }

  document.addEventListener("mouseover", function (event) {
    var row = event.target.closest ? event.target.closest(".row[data-snapshot]") : null;
    if (row && row !== hover.row) showPreview(row, event.clientX, event.clientY);
  });

  document.addEventListener("mousemove", function (event) {
    hover.x = event.clientX;
    hover.y = event.clientY;

    // Work out what is under the cursor on every move rather than relying on
    // mouseover alone. The list re-renders every poll, which swaps the row out
    // from under a stationary cursor without firing a fresh mouseover.
    var under = event.target.closest ? event.target.closest(".row[data-snapshot]") : null;
    if (under && under !== hover.row) {
      showPreview(under, event.clientX, event.clientY);
      return;
    }
    if (!hover.row) return;

    positionPreview(event.clientX, event.clientY);
    if (distanceToRect(event.clientX, event.clientY,
                       hover.row.getBoundingClientRect()) > HIDE_DISTANCE) {
      hidePreview();
    }
  }, { passive: true });

  // Scrolling moves the row out from under the cursor; re-check the distance.
  window.addEventListener("scroll", function () {
    if (!hover.row) return;
    if (distanceToRect(hover.x, hover.y,
                       hover.row.getBoundingClientRect()) > HIDE_DISTANCE) {
      hidePreview();
    }
  }, { passive: true });

  document.addEventListener("mouseleave", hidePreview);

  // Keyboard users get the same preview, anchored beside the focused row.
  document.addEventListener("focusin", function (event) {
    var row = event.target.closest ? event.target.closest(".row[data-snapshot]") : null;
    if (!row) return;
    var rect = row.getBoundingClientRect();
    showPreview(row, rect.right - 40, rect.bottom);
  });
  document.addEventListener("focusout", function (event) {
    if (event.target.classList && event.target.classList.contains("row")) hidePreview();
  });

  /* ------------------------------------------------------------- lightbox */

  function openLightbox(row) {
    var snapshot = row.getAttribute("data-snapshot");
    if (!snapshot) return;
    els.lightboxImg.setAttribute("src", snapshot);
    els.lightboxCaption.textContent = row.getAttribute("data-caption") || "";
    els.lightbox.hidden = false;
  }

  els.log.addEventListener("click", function (event) {
    var row = event.target.closest(".row[data-snapshot]");
    if (row) openLightbox(row);
  });

  els.lightbox.addEventListener("click", function () { els.lightbox.hidden = true; });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { els.lightbox.hidden = true; hidePreview(); }
    if (event.key === "Enter") {
      var row = document.activeElement;
      if (row && row.classList && row.classList.contains("row")) openLightbox(row);
    }
  });

  /* -------------------------------------------------------------- controls */

  els.categories.addEventListener("click", function (event) {
    var button = event.target.closest(".cat");
    if (!button) return;
    state.category = button.getAttribute("data-category") || "";
    els.categories.querySelectorAll(".cat").forEach(function (el) {
      el.classList.toggle("active", (el.getAttribute("data-category") || "") === state.category);
    });
    render([]);
  });

  els.search.addEventListener("input", function () {
    state.query = els.search.value.trim().toLowerCase();
    render([]);
  });

  els.groupToggle.addEventListener("change", function () { render([]); });

  els.togglePause.addEventListener("click", function () {
    state.paused = !state.paused;
    els.togglePause.textContent = state.paused ? "Resume updates" : "Pause updates";
    els.togglePause.classList.toggle("active", state.paused);
    if (!state.paused) poll();
  });

  els.clearLog.addEventListener("click", function () {
    if (!window.confirm("Delete every logged sighting and its snapshot?")) return;
    fetch("/api/clear", { method: "POST" }).then(function () {
      state.entries.clear();
      state.knownCategories.clear();
      state.maxId = 0;
      state.sinceTs = 0;
      state.oldestId = null;
      state.firstLoad = true;
      hidePreview();
      render([]);
      poll();
    });
  });

  els.loadMore.addEventListener("click", function () {
    if (state.oldestId == null) return;
    fetch("/api/log?before_id=" + state.oldestId + "&limit=200")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.entries || !data.entries.length) {
          els.loadMore.textContent = "That is the whole log";
          els.loadMore.disabled = true;
          return;
        }
        data.entries.forEach(function (entry) {
          state.entries.set(entry.id, entry);
          if (state.oldestId === null || entry.id < state.oldestId) state.oldestId = entry.id;
        });
        render([]);
      });
  });

  if (els.toggleStream) {
    els.toggleStream.addEventListener("click", function () {
      var showing = els.streamWrap.hidden;
      els.streamWrap.hidden = !showing;
      els.toggleStream.classList.toggle("active", showing);
      // Only hold the MJPEG connection open while the panel is visible.
      els.streamImg.src = showing ? "/stream.mjpg?t=" + Date.now() : "";
    });
  }

  /* ------------------------------------------------------------------ boot */

  poll();
  setInterval(poll, POLL_MS);
  // Keep relative timestamps ("3m ago") honest between polls.
  setInterval(function () { if (!state.paused) render([]); }, 30000);
})();
