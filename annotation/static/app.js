const MAX_SELECTED = 10;

let currentIdx = 0;
let totalImages = 0;
let annotatedCount = 0;

// ---- API helpers ----

async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    return res.json();
}

// ---- Toast ----

function showToast(msg, duration = 1500) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), duration);
}

// ---- Render ----

function getCheckedLabels() {
    const checked = document.querySelectorAll('.label-row input[type="checkbox"]:checked');
    return Array.from(checked).map(cb => cb.dataset.labelKey + ":" + cb.dataset.labelValue);
}

function getCheckedCount() {
    return document.querySelectorAll('.label-row input[type="checkbox"]:checked').length;
}

function updateSelectionBadge() {
    const count = getCheckedCount();
    const badge = document.getElementById("selection-count");
    badge.textContent = `Выбрано: ${count} / ${MAX_SELECTED}`;
    badge.classList.toggle("full", count >= MAX_SELECTED);
    badge.classList.toggle("has-selection", count > 0 && count < MAX_SELECTED);

    // Disable unchecked checkboxes when limit reached
    document.querySelectorAll('.label-row input[type="checkbox"]').forEach(cb => {
        const row = cb.closest(".label-row");
        if (!cb.checked && count >= MAX_SELECTED) {
            cb.disabled = true;
            row.classList.add("disabled");
        } else {
            cb.disabled = false;
            row.classList.remove("disabled");
        }
    });
}

function renderCategories(categories, savedLabels) {
    const container = document.getElementById("categories");
    container.innerHTML = "";

    const savedSet = new Set(savedLabels || []);

    for (const cat of categories) {
        const section = document.createElement("div");
        section.className = "category";
        section.dataset.cat = cat.key;

        const header = document.createElement("div");
        header.className = "category-header";
        header.innerHTML = `<span>${cat.name} (${cat.labels.length})</span><span class="arrow">&#9660;</span>`;

        const body = document.createElement("div");
        body.className = "category-body";

        header.addEventListener("click", () => {
            body.classList.toggle("hidden");
            header.querySelector(".arrow").classList.toggle("collapsed");
        });

        for (const label of cat.labels) {
            const row = document.createElement("div");
            row.className = "label-row";

            const isChecked = savedSet.has(label.key + ":" + label.value);

            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.dataset.labelKey = label.key;
            cb.dataset.labelValue = label.value;
            cb.checked = isChecked;
            if (isChecked) row.classList.add("checked");

            cb.addEventListener("change", () => {
                row.classList.toggle("checked", cb.checked);
                updateSelectionBadge();
            });

            row.addEventListener("click", (e) => {
                if (e.target === cb) return;
                if (cb.disabled) return;
                cb.checked = !cb.checked;
                row.classList.toggle("checked", cb.checked);
                updateSelectionBadge();
            });

            const desc = document.createElement("span");
            desc.className = "label-desc";
            desc.textContent = label.description;

            const val = document.createElement("span");
            val.className = "label-value";
            val.textContent = label.value;

            row.appendChild(cb);
            row.appendChild(desc);
            row.appendChild(val);
            body.appendChild(row);
        }

        section.appendChild(header);
        section.appendChild(body);
        container.appendChild(section);
    }

    updateSelectionBadge();
}

async function loadImage(idx) {
    const data = await fetchJSON(`/api/image/${idx}`);
    if (data.error) {
        showToast(data.error);
        return;
    }

    currentIdx = data.index;

    // Update image
    document.getElementById("image").src = `/api/image/${idx}/file`;
    const filename = data.image_path.split("/").pop().split("\\").pop();
    const annotatedMark = data.is_annotated ? '<span class="annotated-badge">размечено</span>' : '';
    document.getElementById("image-name").innerHTML = filename + annotatedMark;

    // Update progress
    document.getElementById("progress").textContent =
        `Изображение ${idx + 1} / ${totalImages} (размечено: ${annotatedCount})`;

    // Render labels
    renderCategories(data.categories, data.saved_labels);
}

// ---- Actions ----

async function saveAndNext() {
    const labels = getCheckedLabels();
    if (labels.length === 0) {
        showToast("Выберите хотя бы один признак");
        return;
    }

    const res = await fetchJSON(`/api/image/${currentIdx}/save`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({important_labels: labels}),
    });

    if (res.saved) {
        annotatedCount++;
        showToast(`Сохранено (${labels.length} признаков)`);
        if (res.next_index !== null) {
            loadImage(res.next_index);
        } else {
            showToast("Все изображения размечены!", 3000);
        }
    }
}

function goBack() {
    if (currentIdx > 0) {
        loadImage(currentIdx - 1);
    }
}

function skip() {
    if (currentIdx < totalImages - 1) {
        loadImage(currentIdx + 1);
    }
}

// ---- Init ----

async function init() {
    const status = await fetchJSON("/api/status");
    totalImages = status.total;
    annotatedCount = status.annotated;

    const startIdx = status.first_unannotated;
    loadImage(startIdx);
}

// Buttons
document.getElementById("btn-save").addEventListener("click", saveAndNext);
document.getElementById("btn-back").addEventListener("click", goBack);
document.getElementById("btn-skip").addEventListener("click", skip);

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
    // Ignore if focused on input
    if (e.target.tagName === "INPUT") return;

    if (e.key === "Enter" || (e.ctrlKey && e.key === "s")) {
        e.preventDefault();
        saveAndNext();
    } else if (e.key === "ArrowRight") {
        skip();
    } else if (e.key === "ArrowLeft") {
        goBack();
    }
});

init();
