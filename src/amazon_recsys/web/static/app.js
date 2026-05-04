(() => {
  const workspaceSelector = ".workspace-main";
  const enhancedLinkSelector = "a[data-enhanced-nav]";
  const enhancedFormSelector = "form[data-enhanced-form]";
  const shutdownSelector = "[data-local-shutdown]";
  const tabButtonSelector = "[data-tab-target]";
  const tabLinkSelector = "[data-tab-link]";
  const navItemSelector = ".nav-item[data-tab-link]";
  const tabPanelSelector = "[data-tab-panel]";
  const themeToggleSelector = "[data-theme-toggle]";
  const viewportModeSwitchSelector = "[data-viewport-mode-switch]";
  const viewportModeOptionSelector = "[data-viewport-mode-option]";
  const datasetModalSelector = "[data-dataset-modal]";
  const datasetDismissSelector = "[data-dataset-dismiss]";
  const datasetCopyCommandSelector = "[data-dataset-copy-command]";
  const datasetCommandTextSelector = "[data-dataset-command-text]";
  const localDatasetDownloadSelector = "[data-local-dataset-download]";
  const datasetDownloadStatusSelector = "[data-dataset-download-status]";
  const tabStatePrefix = "amazon-recsys-tab:";
  const tableResizeStoragePrefix = "amazon-recsys-column-widths:";
  const themeStorageKey = "amazon-recsys-theme";
  const viewportModeStorageKey = "amazon-recsys-viewport-mode";
  const datasetPromptStorageKey = "amazon-recsys-dataset-prompt-dismissed";
  const themeMediaQuery = window.matchMedia?.("(prefers-color-scheme: dark)");

  function readStoredTheme() {
    try {
      const storedTheme = window.localStorage.getItem(themeStorageKey);
      return storedTheme === "light" || storedTheme === "dark" ? storedTheme : null;
    } catch (error) {
      return null;
    }
  }

  function writeStoredTheme(theme) {
    try {
      window.localStorage.setItem(themeStorageKey, theme);
    } catch (error) {
      // Keep the selected theme for this page view when storage is unavailable.
    }
  }

  function getSystemTheme() {
    return themeMediaQuery?.matches ? "dark" : "light";
  }

  function getActiveTheme() {
    const documentTheme = document.documentElement.dataset.theme;
    if (documentTheme === "light" || documentTheme === "dark") {
      return documentTheme;
    }
    return readStoredTheme() || getSystemTheme();
  }

  function updateThemeToggle(activeTheme, isManualPreference) {
    const toggle = document.querySelector(themeToggleSelector);
    if (!toggle) {
      return;
    }

    const nextTheme = activeTheme === "dark" ? "light" : "dark";
    const label = toggle.querySelector("[data-theme-toggle-label]");
    const note = toggle.querySelector("[data-theme-toggle-note]");
    const actionLabel = `Switch to ${nextTheme} mode`;

    toggle.dataset.activeTheme = activeTheme;
    toggle.setAttribute("aria-label", actionLabel);
    toggle.setAttribute("title", actionLabel);
    toggle.setAttribute("aria-pressed", String(activeTheme === "dark"));
    if (label) {
      label.textContent = activeTheme === "dark" ? "Dark mode" : "Light mode";
    }
    if (note) {
      note.textContent = isManualPreference ? "Saved preference" : "Follows system";
    }
  }

  function applyTheme(theme, { persist = false } = {}) {
    const nextTheme = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = nextTheme;
    if (persist) {
      writeStoredTheme(nextTheme);
    }
    updateThemeToggle(nextTheme, persist || Boolean(readStoredTheme()));
  }

  function initialiseTheme() {
    applyTheme(readStoredTheme() || getSystemTheme());

    const syncSystemTheme = () => {
      if (!readStoredTheme()) {
        applyTheme(getSystemTheme());
      }
    };
    if (themeMediaQuery?.addEventListener) {
      themeMediaQuery.addEventListener("change", syncSystemTheme);
    } else if (themeMediaQuery?.addListener) {
      themeMediaQuery.addListener(syncSystemTheme);
    }
  }

  function isValidViewportMode(mode) {
    return mode === "desktop" || mode === "mobile";
  }

  function readStoredViewportMode() {
    try {
      const storedMode = window.localStorage.getItem(viewportModeStorageKey);
      return isValidViewportMode(storedMode) ? storedMode : null;
    } catch (error) {
      return null;
    }
  }

  function writeStoredViewportMode(mode) {
    try {
      window.localStorage.setItem(viewportModeStorageKey, mode);
    } catch (error) {
      // Keep the selected viewport for this page view when storage is unavailable.
    }
  }

  function getActiveViewportMode() {
    const documentMode = document.documentElement.dataset.viewportMode;
    if (isValidViewportMode(documentMode)) {
      return documentMode;
    }
    return readStoredViewportMode() || "desktop";
  }

  function updateViewportModeSwitch(activeMode) {
    const switchNode = document.querySelector(viewportModeSwitchSelector);
    const options = Array.from(document.querySelectorAll(viewportModeOptionSelector));
    if (!switchNode || !options.length) {
      return;
    }

    switchNode.dataset.activeViewportMode = activeMode;
    switchNode.setAttribute("aria-label", `Viewport mode: ${activeMode}`);
    options.forEach((option) => {
      const optionMode = option.dataset.viewportModeOption || "";
      const isActive = optionMode === activeMode;
      option.classList.toggle("is-active", isActive);
      option.setAttribute("aria-pressed", String(isActive));
      option.setAttribute("title", `Show ${optionMode} view`);
    });
  }

  function applyViewportMode(mode, { persist = false } = {}) {
    const nextMode = isValidViewportMode(mode) ? mode : "desktop";
    document.documentElement.dataset.viewportMode = nextMode;
    if (persist) {
      writeStoredViewportMode(nextMode);
    }
    updateViewportModeSwitch(nextMode);
  }

  function initialiseViewportMode() {
    applyViewportMode(readStoredViewportMode() || getActiveViewportMode());
  }

  function readDatasetPromptDismissed() {
    try {
      return window.localStorage.getItem(datasetPromptStorageKey) === "true";
    } catch (error) {
      return false;
    }
  }

  function writeDatasetPromptDismissed() {
    try {
      window.localStorage.setItem(datasetPromptStorageKey, "true");
    } catch (error) {
      // The prompt remains dismissible for the current page when storage is unavailable.
    }
  }

  function getDatasetModal() {
    return document.querySelector(datasetModalSelector);
  }

  function setDatasetModalStatus(message, tone = "info") {
    const status = document.querySelector(datasetDownloadStatusSelector);
    if (!status) {
      return;
    }
    status.hidden = false;
    status.textContent = message;
    status.dataset.statusTone = tone;
  }

  function closeDatasetModal({ persist = true } = {}) {
    const modal = getDatasetModal();
    if (!modal) {
      return;
    }
    modal.hidden = true;
    document.body.classList.remove("is-modal-open");
    if (persist) {
      writeDatasetPromptDismissed();
    }
  }

  function openDatasetModal() {
    const modal = getDatasetModal();
    if (!modal) {
      return;
    }
    modal.hidden = false;
    document.body.classList.add("is-modal-open");
    const focusTarget = modal.querySelector(`${localDatasetDownloadSelector}, [data-dataset-manual-link], ${datasetDismissSelector}`);
    focusTarget?.focus?.();
  }

  function initialiseDatasetPrompt() {
    const modal = getDatasetModal();
    if (!modal || readDatasetPromptDismissed()) {
      return;
    }
    openDatasetModal();
  }

  async function copyDatasetCommand(button) {
    const command = button.dataset.command || document.querySelector(datasetCommandTextSelector)?.textContent?.trim() || "";
    if (!command) {
      setDatasetModalStatus("No script command is available for this session.", "warn");
      return;
    }

    try {
      await navigator.clipboard.writeText(command);
      setDatasetModalStatus("Script command copied.", "ready");
    } catch (error) {
      setDatasetModalStatus("Copy failed. Select the command text manually.", "warn");
    }
  }

  async function requestDatasetDownload(button) {
    const endpoint = button.dataset.downloadEndpoint || "/local/download-dataset";
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Starting...";
    setDatasetModalStatus("Starting dataset download...", "info");

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "X-Requested-With": "fetch",
        },
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Download failed with status ${response.status}`);
      }
      const logPath = payload.log_path ? ` Progress log: ${payload.log_path}` : "";
      setDatasetModalStatus(`${payload.detail || "Dataset download started."}${logPath}`, "ready");
      writeDatasetPromptDismissed();
    } catch (error) {
      setDatasetModalStatus(error.message || "Dataset download could not be started.", "warn");
    } finally {
      button.disabled = false;
      button.textContent = originalLabel;
    }
  }

  function formatLabel(value) {
    return value
      .split(/[_+\s]+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function splitMultiValue(rawValue) {
    return String(rawValue || "")
      .split(/[+,]/)
      .map((part) => part.trim())
      .filter(Boolean);
  }

  function getWorkspace() {
    return document.querySelector(workspaceSelector);
  }

  function getActiveTabName(groupName = "workspace") {
    const activeButton = document.querySelector(`${tabButtonSelector}[data-tab-group-name="${groupName}"].is-active`);
    return activeButton?.dataset.tabTarget || "";
  }

  function isModifiedClick(event) {
    return event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0;
  }

  function readStoredTab(groupName) {
    try {
      return window.sessionStorage.getItem(`${tabStatePrefix}${groupName}`);
    } catch (error) {
      return null;
    }
  }

  function writeStoredTab(groupName, tabName) {
    try {
      window.sessionStorage.setItem(`${tabStatePrefix}${groupName}`, tabName);
    } catch (error) {
      // Ignore storage failures and continue with in-memory behaviour.
    }
  }

  function parseColumnWidths(rawValue) {
    return String(rawValue || "")
      .split(",")
      .map((value) => Number.parseInt(value.trim(), 10))
      .map((value) => (Number.isFinite(value) && value > 0 ? value : null));
  }

  function readStoredColumnWidths(tableName) {
    try {
      const rawValue = window.localStorage.getItem(`${tableResizeStoragePrefix}${tableName}`);
      const parsed = JSON.parse(rawValue || "[]");
      return Array.isArray(parsed) ? parsed.map((value) => Number.parseInt(value, 10)) : [];
    } catch (error) {
      return [];
    }
  }

  function writeStoredColumnWidths(tableName, widths) {
    try {
      window.localStorage.setItem(`${tableResizeStoragePrefix}${tableName}`, JSON.stringify(widths));
    } catch (error) {
      // Ignore storage failures; resizing still works for the current DOM.
    }
  }

  function removeStoredColumnWidths(tableName) {
    try {
      window.localStorage.removeItem(`${tableResizeStoragePrefix}${tableName}`);
    } catch (error) {
      // Ignore storage failures.
    }
  }

  function resolveHashTab(root, groupName) {
    const hashValue = window.location.hash.replace(/^#/, "").trim();
    if (!hashValue) {
      return null;
    }
    const targetPanel = root.querySelector(`${tabPanelSelector}[data-tab-group-name="${groupName}"][data-tab-panel="${hashValue}"]`);
    return targetPanel ? hashValue : null;
  }

  function activateTab(root, groupName, tabName, { updateHash = false, scrollToSection = false } = {}) {
    const buttons = Array.from(root.querySelectorAll(`${tabButtonSelector}[data-tab-group-name="${groupName}"]`));
    const panels = Array.from(root.querySelectorAll(`${tabPanelSelector}[data-tab-group-name="${groupName}"]`));
    if (!buttons.length || !panels.length) {
      return false;
    }

    const hasTarget = panels.some((panel) => panel.dataset.tabPanel === tabName);
    if (!hasTarget) {
      return false;
    }

    buttons.forEach((button) => {
      const isActive = button.dataset.tabTarget === tabName;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-selected", String(isActive));
      button.tabIndex = isActive ? 0 : -1;
    });

    panels.forEach((panel) => {
      const isActive = panel.dataset.tabPanel === tabName;
      panel.hidden = !isActive;
      panel.classList.toggle("is-active", isActive);
    });

    document.querySelectorAll(`${navItemSelector}[data-tab-group-name="${groupName}"]`).forEach((item) => {
      const isActive = item.dataset.tabLink === tabName;
      item.classList.toggle("is-active", isActive);
      if (isActive) {
        item.setAttribute("aria-current", "page");
      } else {
        item.removeAttribute("aria-current");
      }
    });

    writeStoredTab(groupName, tabName);

    if (updateHash) {
      const nextUrl = `${window.location.pathname}${window.location.search}#${tabName}`;
      window.history.replaceState(window.history.state, "", nextUrl);
    }

    if (scrollToSection) {
      const sectionId =
        root.querySelector(`[data-tab-group="${groupName}"]`)?.id ||
        root.querySelector(`${tabLinkSelector}[data-tab-group-name="${groupName}"]`)?.dataset.tabSection;
      if (sectionId) {
        document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    return true;
  }

  function initialiseTabs(root = document) {
    const groups = Array.from(root.querySelectorAll("[data-tab-group]"));
    groups.forEach((group) => {
      const groupName = group.dataset.tabGroup;
      if (!groupName) {
        return;
      }
      const defaultTab = group.dataset.defaultTab || group.querySelector(tabButtonSelector)?.dataset.tabTarget;
      const resolvedTab = resolveHashTab(root, groupName) || readStoredTab(groupName) || defaultTab;
      if (resolvedTab) {
        activateTab(root, groupName, resolvedTab, { updateHash: false, scrollToSection: false });
      }
    });
  }

  async function refreshWorkspace(url, { pushState = true } = {}) {
    const currentWorkspace = getWorkspace();
    if (!currentWorkspace) {
      window.location.assign(url);
      return;
    }

    currentWorkspace.classList.add("is-updating");

    try {
      const response = await fetch(url, {
        headers: {
          "X-Requested-With": "fetch",
        },
      });
      if (!response.ok) {
        throw new Error(`Navigation failed with status ${response.status}`);
      }

      const html = await response.text();
      const parsed = new DOMParser().parseFromString(html, "text/html");
      const nextWorkspace = parsed.querySelector(workspaceSelector);
      if (!nextWorkspace) {
        throw new Error("Updated workspace markup was not found.");
      }

      document.title = parsed.title || document.title;
      currentWorkspace.replaceWith(nextWorkspace);
      initialiseTableFilters(document);
      initialiseSortableTables(document);
      initialiseResizableTables(document);
      initialiseMonitoringViews(document);
      initialiseTabs(document);

      if (pushState) {
        window.history.pushState({}, "", url);
      }
    } catch (error) {
      window.location.assign(url);
    } finally {
      const activeWorkspace = getWorkspace();
      if (activeWorkspace) {
        activeWorkspace.classList.remove("is-updating");
      }
    }
  }

  function buildFormUrl(form) {
    const action = form.getAttribute("action") || window.location.pathname;
    const params = new URLSearchParams();
    const formData = new FormData(form);
    for (const [key, value] of formData.entries()) {
      if (typeof value !== "string") {
        continue;
      }
      params.set(key, value);
    }
    const query = params.toString();
    const baseUrl = query ? `${action}?${query}` : action;
    const activeTab = getActiveTabName();
    return activeTab ? `${baseUrl}#${activeTab}` : baseUrl;
  }

  function buildSelectOptions(select, rows, filterKey) {
    if (!select || select.dataset.optionsBuilt === "true") {
      return;
    }

    const optionValues = new Set();
    rows.forEach((row) => {
      const rawValue = row.dataset[filterKey] || "";
      splitMultiValue(rawValue).forEach((value) => optionValues.add(value));
    });

    Array.from(optionValues)
      .sort((left, right) => left.localeCompare(right))
      .forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = formatLabel(value);
        select.appendChild(option);
      });

    select.dataset.optionsBuilt = "true";
  }

  function initialiseTableFilters(root = document) {
    const tables = root.querySelectorAll("[data-filter-table]");
    tables.forEach((table) => {
      if (table.dataset.filtersReady === "true") {
        return;
      }

      const tableName = table.dataset.filterTable;
      const rows = Array.from(table.querySelectorAll(`[data-table-row="${tableName}"]`));
      const searchInput = root.querySelector(`[data-table-search="${tableName}"]`);
      const selectInputs = Array.from(root.querySelectorAll(`[data-table-select="${tableName}"]`));
      const minInputs = Array.from(root.querySelectorAll(`[data-table-min="${tableName}"]`));
      const countLabel = root.querySelector(`[data-table-count="${tableName}"]`);
      const emptyMessage = root.querySelector(`[data-table-empty="${tableName}"]`);
      const totalRows = rows.length;

      selectInputs.forEach((select) => buildSelectOptions(select, rows, select.dataset.filterKey));

      const applyFilters = () => {
        const searchTerm = (searchInput?.value || "").trim().toLowerCase();
        let visibleCount = 0;

        rows.forEach((row) => {
          const rowSearch = (row.dataset.search || "").toLowerCase();
          const matchesSearch = !searchTerm || rowSearch.includes(searchTerm);

          const matchesSelects = selectInputs.every((select) => {
            const selectedValue = (select.value || "").trim().toLowerCase();
            if (!selectedValue) {
              return true;
            }
            const rowValue = (row.dataset[select.dataset.filterKey] || "").toLowerCase();
            return splitMultiValue(rowValue).includes(selectedValue) || rowValue === selectedValue;
          });

          const matchesMinimums = minInputs.every((input) => {
            const minimumValue = Number.parseFloat(input.value || "0");
            if (Number.isNaN(minimumValue) || minimumValue <= 0) {
              return true;
            }
            const rowValue = Number.parseFloat(row.dataset[input.dataset.filterKey] || "0");
            return rowValue >= minimumValue;
          });

          const visible = matchesSearch && matchesSelects && matchesMinimums;
          row.hidden = !visible;
          if (visible) {
            visibleCount += 1;
          }
        });

        if (countLabel) {
          countLabel.textContent = `${visibleCount} of ${totalRows} shown`;
        }
        if (emptyMessage) {
          emptyMessage.hidden = visibleCount !== 0;
        }
      };

      searchInput?.addEventListener("input", applyFilters);
      selectInputs.forEach((select) => select.addEventListener("change", applyFilters));
      minInputs.forEach((input) => input.addEventListener("input", applyFilters));

      applyFilters();
      table.dataset.filtersReady = "true";
    });
  }

  function initialiseSortableTables(root = document) {
    const tables = root.querySelectorAll("[data-sort-table]");
    tables.forEach((table) => {
      if (table.dataset.sortReady === "true") {
        return;
      }

      const tableName = table.dataset.sortTable;
      const tbody = table.querySelector("tbody");
      const triggers = Array.from(root.querySelectorAll(`[data-sort-trigger="${tableName}"]`));
      if (!tbody || !triggers.length) {
        table.dataset.sortReady = "true";
        return;
      }

      const rows = Array.from(tbody.querySelectorAll(`[data-table-row="${tableName}"]`));
      rows.forEach((row, index) => {
        row.dataset.sortIndex = String(index);
      });

      const applySort = (sortKey, sortType, nextDirection) => {
        const orderedRows = [...rows].sort((left, right) => {
          const leftRaw = left.dataset[sortKey] || "";
          const rightRaw = right.dataset[sortKey] || "";

          if (sortType === "number") {
            const leftValue = Number.parseFloat(leftRaw || "0");
            const rightValue = Number.parseFloat(rightRaw || "0");
            if (leftValue === rightValue) {
              return Number.parseInt(left.dataset.sortIndex || "0", 10) - Number.parseInt(right.dataset.sortIndex || "0", 10);
            }
            return nextDirection === "asc" ? leftValue - rightValue : rightValue - leftValue;
          }

          const comparison = leftRaw.localeCompare(rightRaw, undefined, { numeric: true, sensitivity: "base" });
          if (comparison === 0) {
            return Number.parseInt(left.dataset.sortIndex || "0", 10) - Number.parseInt(right.dataset.sortIndex || "0", 10);
          }
          return nextDirection === "asc" ? comparison : -comparison;
        });

        orderedRows.forEach((row) => tbody.appendChild(row));
        table.dataset.sortKey = sortKey;
        table.dataset.sortDirection = nextDirection;

        triggers.forEach((trigger) => {
          const isActive = trigger.dataset.sortKey === sortKey;
          const direction = isActive ? nextDirection : "none";
          trigger.dataset.sortDirection = direction;
          trigger.classList.toggle("is-active", isActive);
          trigger.setAttribute("aria-sort", direction === "none" ? "none" : direction === "asc" ? "ascending" : "descending");
        });
      };

      triggers.forEach((trigger) => {
        trigger.addEventListener("click", () => {
          const sortKey = trigger.dataset.sortKey || "";
          const sortType = trigger.dataset.sortType || "text";
          const isActive = table.dataset.sortKey === sortKey;
          const nextDirection = isActive && table.dataset.sortDirection === "desc" ? "asc" : "desc";
          applySort(sortKey, sortType, nextDirection);
        });
      });

      const defaultTrigger = triggers.find((trigger) => trigger.dataset.sortDefault === "true");
      if (defaultTrigger) {
        applySort(
          defaultTrigger.dataset.sortKey || "",
          defaultTrigger.dataset.sortType || "text",
          defaultTrigger.dataset.sortDefaultDirection || "desc"
        );
      }

      table.dataset.sortReady = "true";
    });
  }

  function ensureColumnGroup(table, columnCount) {
    let columnGroup = table.querySelector(":scope > colgroup");
    if (!columnGroup) {
      columnGroup = document.createElement("colgroup");
      table.insertBefore(columnGroup, table.firstElementChild);
    }

    while (columnGroup.children.length < columnCount) {
      columnGroup.appendChild(document.createElement("col"));
    }
    while (columnGroup.children.length > columnCount) {
      columnGroup.removeChild(columnGroup.lastElementChild);
    }

    return Array.from(columnGroup.children);
  }

  function applyTableMinimumWidth(table, columns, headers) {
    const totalWidth = columns.reduce((total, column, index) => {
      const columnWidth = Number.parseInt(column.style.width || "0", 10);
      const headerWidth = Math.round(headers[index]?.getBoundingClientRect().width || 0);
      return total + (columnWidth || headerWidth || 0);
    }, 0);
    if (totalWidth > 0) {
      table.style.minWidth = `${totalWidth}px`;
    }
  }

  function initialiseResizableTables(root = document) {
    const tables = Array.from(root.querySelectorAll("table.data-table"));
    tables.forEach((table, tableIndex) => {
      if (table.dataset.resizeReady === "true") {
        return;
      }

      const headers = Array.from(table.querySelectorAll("thead th"));
      if (headers.length < 2) {
        table.dataset.resizeReady = "true";
        return;
      }

      const tableName =
        table.dataset.resizableTable ||
        table.dataset.sortTable ||
        table.dataset.filterTable ||
        `table-${tableIndex}`;
      const columns = ensureColumnGroup(table, headers.length);
      const defaultWidths = parseColumnWidths(table.dataset.defaultWidths);
      const storedWidths = readStoredColumnWidths(tableName);
      table.dataset.resizableTable = tableName;
      table.classList.add("is-resizable-table");

      columns.forEach((column, index) => {
        const width = storedWidths[index] || defaultWidths[index];
        if (width) {
          column.style.width = `${width}px`;
        }
      });
      applyTableMinimumWidth(table, columns, headers);

      const resetWidths = () => {
        removeStoredColumnWidths(tableName);
        columns.forEach((column, index) => {
          const width = defaultWidths[index];
          column.style.width = width ? `${width}px` : "";
        });
        applyTableMinimumWidth(table, columns, headers);
      };

      headers.forEach((header, columnIndex) => {
        header.classList.add("resizable-header");
        const handle = document.createElement("span");
        handle.className = "column-resize-handle";
        handle.title = "Drag to resize this column. Double-click to reset saved widths.";
        handle.setAttribute("aria-hidden", "true");
        header.appendChild(handle);

        handle.addEventListener("dblclick", (event) => {
          event.preventDefault();
          event.stopPropagation();
          resetWidths();
        });

        handle.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          event.stopPropagation();
          const startX = event.clientX;
          const startWidth = Math.round(header.getBoundingClientRect().width);
          table.classList.add("is-resizing-columns");
          handle.setPointerCapture?.(event.pointerId);

          const onPointerMove = (moveEvent) => {
            moveEvent.preventDefault();
            const nextWidth = Math.max(72, startWidth + moveEvent.clientX - startX);
            columns[columnIndex].style.width = `${nextWidth}px`;
            applyTableMinimumWidth(table, columns, headers);
          };

          const onPointerUp = () => {
            table.classList.remove("is-resizing-columns");
            document.removeEventListener("pointermove", onPointerMove);
            const widths = headers.map((item) => Math.round(item.getBoundingClientRect().width));
            writeStoredColumnWidths(tableName, widths);
          };

          document.addEventListener("pointermove", onPointerMove, { passive: false });
          document.addEventListener("pointerup", onPointerUp, { once: true });
        });
      });

      table.dataset.resizeReady = "true";
    });
  }

  function initialiseMonitoringViews(root = document) {
    const shells = root.querySelectorAll("[data-monitoring-shell]");
    shells.forEach((shell) => {
      if (shell.dataset.monitoringReady === "true") {
        return;
      }

      const panels = Array.from(shell.querySelectorAll("[data-monitoring-window]"));
      const selector = shell.querySelector("[data-monitoring-window-select]");
      const toggleButtons = Array.from(shell.querySelectorAll("[data-monitoring-compare]"));
      const quickJumpSets = Array.from(shell.querySelectorAll("[data-monitoring-jump-set]"));
      if (!panels.length) {
        shell.dataset.monitoringReady = "true";
        return;
      }

      const activateWindow = (windowId) => {
        panels.forEach((panel) => {
          const isActive = panel.dataset.monitoringWindow === windowId;
          panel.hidden = !isActive;
          panel.classList.toggle("is-active", isActive);
        });
        quickJumpSets.forEach((setNode) => {
          const isActive = setNode.dataset.monitoringJumpSet === windowId;
          setNode.hidden = !isActive;
        });
        if (selector && selector.value !== windowId) {
          selector.value = windowId;
        }
      };

      selector?.addEventListener("change", () => {
        activateWindow(selector.value);
      });

      toggleButtons.forEach((button) => {
        button.addEventListener("click", () => {
          const nextMode = button.dataset.monitoringCompare || "baseline";
          shell.dataset.compareMode = nextMode;
          toggleButtons.forEach((candidate) => {
            const isActive = candidate === button;
            candidate.classList.toggle("is-active", isActive);
            candidate.setAttribute("aria-pressed", String(isActive));
          });
        });
      });

      const defaultWindow = selector?.value || shell.dataset.defaultWindow || panels[panels.length - 1].dataset.monitoringWindow;
      activateWindow(defaultWindow || panels[0].dataset.monitoringWindow || "");
      shell.dataset.monitoringReady = "true";
    });
  }

  async function requestLocalShutdown(button) {
    const confirmed = window.confirm("Shutdown the local FastAPI server now?");
    if (!confirmed) {
      return;
    }

    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Shutting down...";

    try {
      const response = await fetch("/local/shutdown", {
        method: "POST",
        headers: {
          "X-Requested-With": "fetch",
        },
      });
      if (!response.ok) {
        throw new Error(`Shutdown failed with status ${response.status}`);
      }
      button.textContent = "Server stopping...";
    } catch (error) {
      button.disabled = false;
      button.textContent = originalLabel;
      window.alert("Local shutdown failed. Check that the app is running in the local environment.");
    }
  }

  document.addEventListener("click", (event) => {
    const themeToggle = event.target.closest(themeToggleSelector);
    if (themeToggle) {
      event.preventDefault();
      const nextTheme = getActiveTheme() === "dark" ? "light" : "dark";
      applyTheme(nextTheme, { persist: true });
      return;
    }

    const viewportModeOption = event.target.closest(viewportModeOptionSelector);
    if (viewportModeOption) {
      event.preventDefault();
      applyViewportMode(viewportModeOption.dataset.viewportModeOption, { persist: true });
      return;
    }

    const datasetDismiss = event.target.closest(datasetDismissSelector);
    if (datasetDismiss) {
      event.preventDefault();
      closeDatasetModal();
      return;
    }

    const datasetCopyCommand = event.target.closest(datasetCopyCommandSelector);
    if (datasetCopyCommand) {
      event.preventDefault();
      void copyDatasetCommand(datasetCopyCommand);
      return;
    }

    const localDatasetDownload = event.target.closest(localDatasetDownloadSelector);
    if (localDatasetDownload) {
      event.preventDefault();
      void requestDatasetDownload(localDatasetDownload);
      return;
    }

    const tabButton = event.target.closest(tabButtonSelector);
    if (tabButton) {
      event.preventDefault();
      activateTab(
        document,
        tabButton.dataset.tabGroupName || "",
        tabButton.dataset.tabTarget || "",
        { updateHash: true, scrollToSection: false }
      );
      return;
    }

    const tabLink = event.target.closest(tabLinkSelector);
    if (tabLink) {
      event.preventDefault();
      activateTab(
        document,
        tabLink.dataset.tabGroupName || "",
        tabLink.dataset.tabLink || "",
        { updateHash: true, scrollToSection: true }
      );
      return;
    }

    const shutdownButton = event.target.closest(shutdownSelector);
    if (shutdownButton) {
      event.preventDefault();
      void requestLocalShutdown(shutdownButton);
      return;
    }

    const link = event.target.closest(enhancedLinkSelector);
    if (!link) {
      return;
    }
    if (isModifiedClick(event)) {
      return;
    }

    const href = link.getAttribute("href");
    if (!href || href.startsWith("#")) {
      return;
    }

    event.preventDefault();
    const resolvedUrl = new URL(link.href, window.location.origin);
    if (!resolvedUrl.hash) {
      const activeTab = getActiveTabName();
      if (activeTab && resolvedUrl.pathname === window.location.pathname) {
        resolvedUrl.hash = activeTab;
      }
    }
    void refreshWorkspace(resolvedUrl.toString(), { pushState: true });
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest(enhancedFormSelector);
    if (!form) {
      return;
    }

    event.preventDefault();
    const submitter = event.submitter;
    const originalLabel = submitter && "textContent" in submitter ? submitter.textContent : null;
    if (submitter && "disabled" in submitter) {
      submitter.disabled = true;
      if ("textContent" in submitter) {
        submitter.textContent = "Running...";
      }
    }

    const url = buildFormUrl(form);
    void refreshWorkspace(url, { pushState: true }).finally(() => {
      if (submitter && "disabled" in submitter) {
        submitter.disabled = false;
      }
      if (submitter && originalLabel !== null && "textContent" in submitter) {
        submitter.textContent = originalLabel;
      }
    });
  });

  window.addEventListener("popstate", () => {
    void refreshWorkspace(`${window.location.pathname}${window.location.search}${window.location.hash}`, { pushState: false });
  });

  window.addEventListener("hashchange", () => {
    const matchedButton = document.querySelector(`${tabButtonSelector}[data-tab-target="${window.location.hash.replace(/^#/, "")}"]`);
    if (!matchedButton) {
      return;
    }
    activateTab(
      document,
      matchedButton.dataset.tabGroupName || "",
      matchedButton.dataset.tabTarget || "",
      { updateHash: false, scrollToSection: false }
    );
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && getDatasetModal() && !getDatasetModal().hidden) {
      closeDatasetModal();
    }
  });

  initialiseTheme();
  initialiseViewportMode();
  initialiseDatasetPrompt();
  initialiseTableFilters(document);
  initialiseSortableTables(document);
  initialiseResizableTables(document);
  initialiseMonitoringViews(document);
  initialiseTabs(document);
})();
