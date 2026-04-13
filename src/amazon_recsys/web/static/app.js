(() => {
  const workspaceSelector = ".workspace-main";
  const enhancedLinkSelector = "a[data-enhanced-nav]";
  const enhancedFormSelector = "form[data-enhanced-form]";
  const shutdownSelector = "[data-local-shutdown]";

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

  function isModifiedClick(event) {
    return event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0;
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
    return query ? `${action}?${query}` : action;
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
    void refreshWorkspace(link.href, { pushState: true });
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
    void refreshWorkspace(`${window.location.pathname}${window.location.search}`, { pushState: false });
  });

  initialiseTableFilters(document);
})();
