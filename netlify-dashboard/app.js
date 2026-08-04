/* =================================================================
   LINKEDIN JOB MARKET ANALYTICS — MAIN APPLICATION JS
   =================================================================
   Handles: Navigation, data loading, Plotly chart rendering,
   EDA filtering, salary estimation lookup.
   ================================================================= */

// ─── Plotly Theme ────────────────────────────────────────────────
const PLOTLY_LAYOUT = {
    template: 'plotly_dark',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8' },
    margin: { l: 50, r: 30, t: 30, b: 50 },
    xaxis: { gridcolor: 'rgba(100,116,175,0.1)', zerolinecolor: 'rgba(100,116,175,0.15)' },
    yaxis: { gridcolor: 'rgba(100,116,175,0.1)', zerolinecolor: 'rgba(100,116,175,0.15)' },
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false,
};

const COLORS = {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#34d399',
    danger: '#f87171',
    warning: '#fbbf24',
    palette: ['#667eea', '#34d399', '#f87171', '#fbbf24', '#a78bfa', '#f472b6', '#38bdf8', '#fb923c'],
    set2: ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'],
    viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
};

// ─── Data Cache ──────────────────────────────────────────────────
const dataCache = {};

async function loadJSON(filename) {
    if (dataCache[filename]) return dataCache[filename];
    try {
        const resp = await fetch(`data/${filename}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        dataCache[filename] = data;
        return data;
    } catch (err) {
        console.error(`Failed to load ${filename}:`, err);
        return null;
    }
}

// ─── Utility Functions ───────────────────────────────────────────
function fmt(n) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    return n.toLocaleString('en-US');
}

function fmtDollar(n) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    return '$' + Math.round(n).toLocaleString('en-US');
}

function median(arr) {
    if (!arr.length) return 0;
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function quantile(arr, q) {
    if (!arr.length) return 0;
    const s = [...arr].sort((a, b) => a - b);
    const pos = (s.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (s[base + 1] !== undefined) {
        return s[base] + rest * (s[base + 1] - s[base]);
    }
    return s[base];
}

function std(arr) {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((sum, val) => sum + (val - m) ** 2, 0) / (arr.length - 1));
}

// ─── Navigation ──────────────────────────────────────────────────
const navItems = document.querySelectorAll('.nav-item');
const pages = document.querySelectorAll('.page');
const sidebar = document.getElementById('sidebar');
const menuToggle = document.getElementById('menuToggle');

const pageInitialized = {};

navItems.forEach(item => {
    item.addEventListener('click', () => {
        const pageName = item.dataset.page;
        navItems.forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        pages.forEach(p => p.classList.remove('active'));
        document.getElementById(`page-${pageName}`).classList.add('active');

        // Close mobile sidebar
        sidebar.classList.remove('open');

        // Initialize page data on first visit
        if (!pageInitialized[pageName]) {
            pageInitialized[pageName] = true;
            initPage(pageName);
        }
    });
});

menuToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
});

// Close sidebar on outside click (mobile)
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 900 && !sidebar.contains(e.target) && e.target !== menuToggle) {
        sidebar.classList.remove('open');
    }
});

// ─── Tab Navigation ──────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        const parent = btn.closest('.page') || document;
        parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(tabId).classList.add('active');
    });
});

// ─── Page Initializer ────────────────────────────────────────────
function initPage(pageName) {
    switch (pageName) {
        case 'overview': initOverview(); break;
        case 'eda': initEDA(); break;
        case 'model': initModel(); break;
        case 'predictor': initPredictor(); break;
        case 'report': break; // Static content, no init needed
    }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: OVERVIEW
// ═══════════════════════════════════════════════════════════════════
async function initOverview() {
    const data = await loadJSON('overview.json');
    if (!data) return;

    // KPIs
    const k = data.kpis;
    document.getElementById('kpi-total').textContent = fmt(k.total_postings);
    document.getElementById('kpi-salary').textContent = fmtDollar(k.avg_salary);
    document.getElementById('kpi-categories').textContent = k.n_categories;
    document.getElementById('kpi-remote').textContent = k.remote_pct.toFixed(1) + '%';

    // Salary Distribution Histogram
    if (data.salary_histogram && data.salary_histogram.length) {
        const medianVal = k.median_salary;
        Plotly.newPlot('chart-salary-dist', [{
            x: data.salary_histogram,
            type: 'histogram',
            nbinsx: 50,
            marker: { color: COLORS.primary, opacity: 0.8 },
            hovertemplate: 'Range: %{x}<br>Count: %{y}<extra></extra>',
        }], {
            ...PLOTLY_LAYOUT,
            xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Annual Salary ($)' },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Count' },
            shapes: [{
                type: 'line', x0: medianVal, x1: medianVal, y0: 0, y1: 1,
                yref: 'paper', line: { color: COLORS.danger, width: 2, dash: 'dash' },
            }],
            annotations: [{
                x: medianVal, y: 1, yref: 'paper', text: `Median: ${fmtDollar(medianVal)}`,
                showarrow: true, arrowhead: 0, ax: 60, ay: -20,
                font: { color: COLORS.danger, size: 12 },
            }],
        }, PLOTLY_CONFIG);
    }

    // Work Type Pie
    if (data.work_type && data.work_type.labels) {
        Plotly.newPlot('chart-work-type', [{
            labels: data.work_type.labels,
            values: data.work_type.values,
            type: 'pie',
            hole: 0.45,
            marker: { colors: COLORS.set2 },
            textinfo: 'label+percent',
            textposition: 'inside',
            hovertemplate: '%{label}<br>%{value:,} postings<br>%{percent}<extra></extra>',
        }], {
            ...PLOTLY_LAYOUT,
            showlegend: true,
            legend: { font: { size: 11 } },
        }, PLOTLY_CONFIG);
    }

    // Category Salary Bar
    if (data.category_salary && data.category_salary.categories) {
        const cs = data.category_salary;
        Plotly.newPlot('chart-category-salary', [{
            y: cs.categories,
            x: cs.medians,
            type: 'bar',
            orientation: 'h',
            marker: {
                color: cs.medians,
                colorscale: 'Viridis',
                showscale: false,
            },
            text: cs.medians.map(v => fmtDollar(v)),
            textposition: 'outside',
            hovertemplate: '%{y}<br>Median: %{text}<br>Count: %{customdata}<extra></extra>',
            customdata: cs.counts,
        }], {
            ...PLOTLY_LAYOUT,
            xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Median Salary ($)' },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, automargin: true },
            margin: { ...PLOTLY_LAYOUT.margin, l: 150 },
        }, PLOTLY_CONFIG);
    }

    // Experience Level Box Plot
    if (data.experience_boxplot && data.experience_boxplot.length) {
        const traces = data.experience_boxplot.map((d, i) => ({
            type: 'box',
            name: d.level,
            lowerfence: [d.min],
            q1: [d.q1],
            median: [d.median],
            q3: [d.q3],
            upperfence: [d.max],
            marker: { color: COLORS.set2[i % COLORS.set2.length] },
            hovertemplate: `${d.level}<br>Median: ${fmtDollar(d.median)}<br>Q1: ${fmtDollar(d.q1)}<br>Q3: ${fmtDollar(d.q3)}<extra></extra>`,
        }));
        Plotly.newPlot('chart-exp-salary', traces, {
            ...PLOTLY_LAYOUT,
            showlegend: false,
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Annual Salary ($)' },
        }, PLOTLY_CONFIG);
    }
}


// ═══════════════════════════════════════════════════════════════════
// PAGE: EDA EXPLORER
// ═══════════════════════════════════════════════════════════════════
let edaData = null;

async function initEDA() {
    edaData = await loadJSON('eda_sample.json');
    if (!edaData) return;

    // Populate filter options
    const categories = [...new Set(edaData.map(d => d.title_category))].filter(Boolean).sort();
    const catSelect = document.getElementById('filter-category');
    categories.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        catSelect.appendChild(opt);
    });

    const experiences = [...new Set(edaData.map(d => d.formatted_experience_level))].filter(Boolean).sort();
    const expSelect = document.getElementById('filter-experience');
    experiences.forEach(e => {
        const opt = document.createElement('option');
        opt.value = e;
        opt.textContent = e;
        expSelect.appendChild(opt);
    });

    // Add filter listeners
    catSelect.addEventListener('change', renderEDACharts);
    expSelect.addEventListener('change', renderEDACharts);
    document.getElementById('filter-remote').addEventListener('change', renderEDACharts);

    renderEDACharts();
}

function getFilteredEDA() {
    if (!edaData) return [];
    const cat = document.getElementById('filter-category').value;
    const exp = document.getElementById('filter-experience').value;
    const remote = document.getElementById('filter-remote').value;

    return edaData.filter(d => {
        if (cat !== 'All' && d.title_category !== cat) return false;
        if (exp !== 'All' && d.formatted_experience_level !== exp) return false;
        if (remote === '1' && d.is_remote !== 1) return false;
        if (remote === '0' && d.is_remote !== 0) return false;
        return true;
    });
}

function renderEDACharts() {
    const filtered = getFilteredEDA();
    const total = edaData ? edaData.length : 0;

    document.getElementById('eda-count').innerHTML =
        `📊 Menampilkan <strong>${fmt(filtered.length)}</strong> postingan dari ${fmt(total)} total (sample dengan data gaji)`;

    // 1. Salary Distribution
    const salaries = filtered.map(d => d.annual_salary).filter(v => v > 0);
    if (salaries.length > 0) {
        Plotly.newPlot('chart-eda-salary', [{
            x: salaries,
            type: 'histogram',
            nbinsx: 40,
            marker: { color: COLORS.primary, opacity: 0.8 },
        }], {
            ...PLOTLY_LAYOUT,
            xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Annual Salary ($)' },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Count' },
        }, PLOTLY_CONFIG);
    }

    // 2. Top 10 Locations
    const locMap = {};
    filtered.forEach(d => {
        if (d.job_state && d.job_state !== 'Unknown' && d.job_state !== 'nan') {
            if (!locMap[d.job_state]) locMap[d.job_state] = [];
            locMap[d.job_state].push(d.annual_salary);
        }
    });
    const locData = Object.entries(locMap)
        .filter(([, vals]) => vals.length >= 3)
        .map(([loc, vals]) => ({ loc, median: median(vals), count: vals.length }))
        .sort((a, b) => b.median - a.median)
        .slice(0, 10);

    if (locData.length > 0) {
        Plotly.newPlot('chart-eda-location', [{
            x: locData.map(d => d.loc),
            y: locData.map(d => d.median),
            type: 'bar',
            marker: {
                color: locData.map(d => d.median),
                colorscale: 'RdYlGn',
                showscale: false,
            },
            text: locData.map(d => fmtDollar(d.median)),
            textposition: 'outside',
            hovertemplate: '%{x}<br>Median: %{text}<br>Count: %{customdata}<extra></extra>',
            customdata: locData.map(d => d.count),
        }], {
            ...PLOTLY_LAYOUT,
            xaxis: { ...PLOTLY_LAYOUT.xaxis, tickangle: -45 },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Median Salary ($)' },
        }, PLOTLY_CONFIG);
    }

    // 3. Remote vs On-site
    const remoteJobs = filtered.filter(d => d.is_remote === 1);
    const onsiteJobs = filtered.filter(d => d.is_remote === 0);
    const remoteSalaries = remoteJobs.map(d => d.annual_salary).filter(v => v > 0);
    const onsiteSalaries = onsiteJobs.map(d => d.annual_salary).filter(v => v > 0);

    if (remoteSalaries.length > 0 || onsiteSalaries.length > 0) {
        Plotly.newPlot('chart-eda-remote', [
            {
                x: ['On-site', 'Remote'],
                y: [onsiteSalaries.length ? median(onsiteSalaries) : 0, remoteSalaries.length ? median(remoteSalaries) : 0],
                type: 'bar',
                marker: { color: [COLORS.danger, COLORS.success] },
                text: [
                    onsiteSalaries.length ? fmtDollar(median(onsiteSalaries)) : '—',
                    remoteSalaries.length ? fmtDollar(median(remoteSalaries)) : '—'
                ],
                textposition: 'outside',
                hovertemplate: '%{x}<br>Median: %{text}<br>Count: %{customdata}<extra></extra>',
                customdata: [onsiteSalaries.length, remoteSalaries.length],
            }
        ], {
            ...PLOTLY_LAYOUT,
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Median Salary ($)' },
            showlegend: false,
        }, PLOTLY_CONFIG);
    }

    // 4. Views vs Applies Scatter
    const scatterData = filtered.filter(d => d.views > 0 && d.applies > 0);
    const sampleScatter = scatterData.length > 3000 ? scatterData.slice(0, 3000) : scatterData;

    if (sampleScatter.length > 0) {
        Plotly.newPlot('chart-eda-scatter', [{
            x: sampleScatter.map(d => d.views),
            y: sampleScatter.map(d => d.applies),
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: sampleScatter.map(d => d.annual_salary),
                colorscale: 'Viridis',
                size: 5,
                opacity: 0.5,
                colorbar: { title: 'Salary ($)', thickness: 15 },
            },
            hovertemplate: 'Views: %{x}<br>Applies: %{y}<br>Salary: %{marker.color:$,.0f}<extra></extra>',
        }], {
            ...PLOTLY_LAYOUT,
            xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Views' },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Applications' },
        }, PLOTLY_CONFIG);
    }

    // 5. Descriptive Statistics Table
    renderStatsTable(filtered);
}

function renderStatsTable(data) {
    const container = document.getElementById('eda-stats-table');
    const salaries = data.map(d => d.annual_salary).filter(v => v > 0);
    const views = data.map(d => d.views).filter(v => v != null && !isNaN(v));
    const applies = data.map(d => d.applies).filter(v => v != null && !isNaN(v));

    const stats = [
        { name: 'Count', salary: salaries.length, views: views.length, applies: applies.length },
        { name: 'Mean', salary: mean(salaries), views: mean(views), applies: mean(applies) },
        { name: 'Std', salary: std(salaries), views: std(views), applies: std(applies) },
        { name: 'Min', salary: Math.min(...salaries), views: Math.min(...views), applies: Math.min(...applies) },
        { name: 'Q25', salary: quantile(salaries, 0.25), views: quantile(views, 0.25), applies: quantile(applies, 0.25) },
        { name: 'Median', salary: median(salaries), views: median(views), applies: median(applies) },
        { name: 'Q75', salary: quantile(salaries, 0.75), views: quantile(views, 0.75), applies: quantile(applies, 0.75) },
        { name: 'Max', salary: Math.max(...salaries), views: Math.max(...views), applies: Math.max(...applies) },
    ];

    let html = `<table class="stats-table">
        <thead><tr><th>Statistic</th><th>Annual Salary ($)</th><th>Views</th><th>Applies</th></tr></thead>
        <tbody>`;
    stats.forEach(s => {
        html += `<tr>
            <td>${s.name}</td>
            <td>${s.name === 'Count' ? fmt(s.salary) : fmtDollar(s.salary)}</td>
            <td>${fmt(Math.round(s.views))}</td>
            <td>${fmt(Math.round(s.applies))}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}


// ═══════════════════════════════════════════════════════════════════
// PAGE: MODEL PERFORMANCE
// ═══════════════════════════════════════════════════════════════════
async function initModel() {
    const data = await loadJSON('model.json');
    if (!data) return;

    // ─── Regression Model Cards ─────────────────────────────────
    if (data.regression) {
        const models = Object.keys(data.regression);
        const emojis = ['🌲', '🚀', '📐'];
        const cardsContainer = document.getElementById('model-cards');
        cardsContainer.innerHTML = '';

        models.forEach((name, i) => {
            const m = data.regression[name];
            const r2 = m.R2_test || 0;
            const mae = m.MAE_test || 0;
            cardsContainer.innerHTML += `
                <div class="model-card">
                    <div class="model-name">${emojis[i] || '📊'} ${name}</div>
                    <div class="model-r2-label">R² Score (Test)</div>
                    <div class="model-r2">${r2.toFixed(4)}</div>
                    <div class="model-mae">MAE = ${fmtDollar(mae)}</div>
                </div>`;
        });

        // ─── Metrics Table ──────────────────────────────────────
        const metricKeys = ['MAE_train', 'MAE_test', 'RMSE_train', 'RMSE_test', 'R2_train', 'R2_test', 'MAPE_test'];
        let tableHTML = `<table class="metrics-table"><thead><tr><th>Model</th>`;
        metricKeys.forEach(k => { tableHTML += `<th>${k}</th>`; });
        tableHTML += '</tr></thead><tbody>';

        // Find best values for highlighting
        const bestR2 = Math.max(...models.map(m => data.regression[m].R2_test || 0));
        const bestMAE = Math.min(...models.map(m => data.regression[m].MAE_test || Infinity));
        const bestRMSE = Math.min(...models.map(m => data.regression[m].RMSE_test || Infinity));

        models.forEach(name => {
            const m = data.regression[name];
            tableHTML += `<tr><td class="model-name-cell">${name}</td>`;
            metricKeys.forEach(k => {
                const val = m[k];
                let cls = '';
                if (k === 'R2_test' && val === bestR2) cls = 'highlight-best';
                if (k === 'MAE_test' && val === bestMAE) cls = 'highlight-best';
                if (k === 'RMSE_test' && val === bestRMSE) cls = 'highlight-best';
                tableHTML += `<td class="${cls}">${val !== undefined ? val.toFixed(4) : '—'}</td>`;
            });
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody></table>';
        document.getElementById('regression-table').innerHTML = tableHTML;

        // ─── R² Chart ───────────────────────────────────────────
        const r2Data = [
            {
                x: models, y: models.map(m => data.regression[m].R2_train || 0),
                name: 'Train', type: 'bar', marker: { color: COLORS.primary },
            },
            {
                x: models, y: models.map(m => data.regression[m].R2_test || 0),
                name: 'Test', type: 'bar', marker: { color: COLORS.danger },
            },
        ];
        Plotly.newPlot('chart-r2', r2Data, {
            ...PLOTLY_LAYOUT,
            barmode: 'group',
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'R² Score', range: [0, 1] },
            legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' },
        }, PLOTLY_CONFIG);

        // ─── MAE Chart ──────────────────────────────────────────
        const maeData = [
            {
                x: models, y: models.map(m => data.regression[m].MAE_train || 0),
                name: 'Train', type: 'bar', marker: { color: COLORS.primary },
            },
            {
                x: models, y: models.map(m => data.regression[m].MAE_test || 0),
                name: 'Test', type: 'bar', marker: { color: COLORS.danger },
            },
        ];
        Plotly.newPlot('chart-mae', maeData, {
            ...PLOTLY_LAYOUT,
            barmode: 'group',
            yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Mean Absolute Error ($)' },
            legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' },
        }, PLOTLY_CONFIG);
    }

    // ─── Actual vs Predicted Scatter ────────────────────────────
    if (data.predictions) {
        const scatterModels = [
            { key: 'rf', label: 'Random Forest' },
            { key: 'xgb', label: 'XGBoost' },
            { key: 'ridge', label: 'Ridge Regression' },
        ];

        const selectorContainer = document.getElementById('scatter-model-selector');
        selectorContainer.innerHTML = '';
        scatterModels.forEach((m, i) => {
            const btn = document.createElement('button');
            btn.className = 'model-btn' + (i === 0 ? ' active' : '');
            btn.textContent = m.label;
            btn.addEventListener('click', () => {
                selectorContainer.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                renderScatterPred(data.predictions, m.key, m.label);
            });
            selectorContainer.appendChild(btn);
        });

        renderScatterPred(data.predictions, 'rf', 'Random Forest');
    }

    // ─── Feature Importance ─────────────────────────────────────
    if (data.feature_importance) {
        const fi = data.feature_importance;
        const features = [...fi.features].reverse();
        const importances = [...fi.importances].reverse();

        Plotly.newPlot('chart-feature-imp', [{
            y: features,
            x: importances,
            type: 'bar',
            orientation: 'h',
            marker: {
                color: importances,
                colorscale: 'Viridis',
                showscale: false,
            },
            hovertemplate: '%{y}<br>Importance: %{x:.4f}<extra></extra>',
        }], {
            ...PLOTLY_LAYOUT,
            margin: { ...PLOTLY_LAYOUT.margin, l: 200 },
            yaxis: { ...PLOTLY_LAYOUT.yaxis, automargin: true },
            xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Importance' },
        }, PLOTLY_CONFIG);
    }

    // ─── Classification Metrics Table ───────────────────────────
    if (data.classification) {
        const clsModels = Object.keys(data.classification);
        const clsKeys = ['Accuracy', 'Precision', 'Recall', 'F1_Score'];

        let clsHTML = `<table class="metrics-table"><thead><tr><th>Model</th>`;
        clsKeys.forEach(k => { clsHTML += `<th>${k}</th>`; });
        clsHTML += '</tr></thead><tbody>';

        const bestAcc = Math.max(...clsModels.map(m => data.classification[m].Accuracy || 0));

        clsModels.forEach(name => {
            const m = data.classification[name];
            clsHTML += `<tr><td class="model-name-cell">${name}</td>`;
            clsKeys.forEach(k => {
                const val = m[k];
                let cls = (k === 'Accuracy' && val === bestAcc) ? 'highlight-best' : '';
                clsHTML += `<td class="${cls}">${val !== undefined ? (val * 100).toFixed(2) + '%' : '—'}</td>`;
            });
            clsHTML += '</tr>';
        });
        clsHTML += '</tbody></table>';
        document.getElementById('classification-table').innerHTML = clsHTML;
    }

    // ─── Confusion Matrices ─────────────────────────────────────
    if (data.confusion_matrices && data.confusion_labels) {
        const labels = data.confusion_labels;
        const cmContainer = document.getElementById('confusion-matrices');
        cmContainer.innerHTML = '';

        Object.keys(data.confusion_matrices).forEach(modelName => {
            const cm = data.confusion_matrices[modelName];
            const div = document.createElement('div');
            div.className = 'chart-card';
            div.innerHTML = `<div class="chart-title"><span class="icon">🧩</span> CM: ${modelName}</div>
                <div id="cm-${modelName}" style="height:350px"></div>`;
            cmContainer.appendChild(div);

            setTimeout(() => {
                Plotly.newPlot(`cm-${modelName}`, [{
                    z: cm,
                    x: labels,
                    y: labels,
                    type: 'heatmap',
                    colorscale: 'Blues',
                    showscale: false,
                    text: cm.map(row => row.map(v => v.toString())),
                    texttemplate: '%{text}',
                    textfont: { size: 14, color: '#fff' },
                    hovertemplate: 'Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
                }], {
                    ...PLOTLY_LAYOUT,
                    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Predicted', side: 'bottom' },
                    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Actual', autorange: 'reversed' },
                    margin: { l: 60, r: 20, t: 10, b: 50 },
                }, PLOTLY_CONFIG);
            }, 100);
        });
    }
}

function renderScatterPred(predictions, modelKey, modelLabel) {
    const actual = predictions.actual;
    const predicted = predictions[modelKey];

    const minVal = Math.min(Math.min(...actual), Math.min(...predicted));
    const maxVal = Math.max(Math.max(...actual), Math.max(...predicted));

    Plotly.newPlot('chart-scatter-pred', [
        {
            x: actual, y: predicted,
            mode: 'markers',
            type: 'scatter',
            marker: { size: 4, opacity: 0.35, color: COLORS.primary },
            name: 'Predictions',
            hovertemplate: 'Actual: %{x:$,.0f}<br>Predicted: %{y:$,.0f}<extra></extra>',
        },
        {
            x: [minVal, maxVal], y: [minVal, maxVal],
            mode: 'lines',
            type: 'scatter',
            line: { color: COLORS.danger, dash: 'dash', width: 2 },
            name: 'Perfect Prediction',
        },
    ], {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Actual Salary ($)' },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Predicted Salary ($)' },
        title: { text: `Actual vs Predicted — ${modelLabel}`, font: { size: 14, color: '#e2e8f0' } },
        showlegend: true,
        legend: { orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' },
    }, PLOTLY_CONFIG);
}


// ═══════════════════════════════════════════════════════════════════
// PAGE: SALARY ESTIMATOR
// ═══════════════════════════════════════════════════════════════════
let salaryLookup = null;

async function initPredictor() {
    salaryLookup = await loadJSON('salary_lookup.json');
    if (!salaryLookup) return;

    // Populate category select
    const catSelect = document.getElementById('pred-category');
    if (salaryLookup._categories) {
        salaryLookup._categories.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c;
            opt.textContent = c;
            catSelect.appendChild(opt);
        });
    }

    document.getElementById('predict-btn').addEventListener('click', predictSalary);
}

function predictSalary() {
    if (!salaryLookup) return;

    const exp = document.getElementById('pred-experience').value;
    const cat = document.getElementById('pred-category').value;

    const key = `${cat}|${exp}`;
    let result = salaryLookup[key];

    // Fallback: try category-only or experience-only
    if (!result) result = salaryLookup[`${cat}|All`];
    if (!result) result = salaryLookup[`All|${exp}`];
    if (!result) result = salaryLookup['All|All'];

    if (!result) {
        document.getElementById('prediction-output').innerHTML =
            '<div class="info-badge">❌ Tidak cukup data untuk kombinasi ini.</div>';
        return;
    }

    const globalMedian = salaryLookup['All|All'] ? salaryLookup['All|All'].median : result.median;
    const vsMedian = ((result.median / globalMedian) - 1) * 100;
    const detailText = `${exp !== 'All' ? exp : 'Semua Level'} • ${cat !== 'All' ? cat : 'Semua Kategori'}`;

    document.getElementById('prediction-output').innerHTML = `
        <div class="prediction-result">
            <h2>💰 Estimasi Gaji Tahunan</h2>
            <div class="salary-value">${fmtDollar(result.median)}</div>
            <div class="salary-detail">${detailText}</div>
            <div class="salary-range">Range: ${fmtDollar(result.q25)} — ${fmtDollar(result.q75)}</div>
        </div>
        <div class="section-header">📊 Konteks Pasar</div>
        <div class="context-grid">
            <div class="context-card">
                <div class="context-label">Median Segment</div>
                <div class="context-value">${fmtDollar(result.median)}</div>
            </div>
            <div class="context-card">
                <div class="context-label">vs Median Global</div>
                <div class="context-value" style="color: ${vsMedian >= 0 ? COLORS.success : COLORS.danger}">${vsMedian >= 0 ? '+' : ''}${vsMedian.toFixed(1)}%</div>
            </div>
            <div class="context-card">
                <div class="context-label">Jumlah Data</div>
                <div class="context-value">${fmt(result.count)}</div>
            </div>
        </div>
    `;
}


// ═══════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Overview page on load
    pageInitialized['overview'] = true;
    initOverview();
});
