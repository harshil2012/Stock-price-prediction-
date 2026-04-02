const API_BASE_URL = 'http://localhost:8000';

let stockChart = null;
let currentCurrencySymbol = '$';

function getCurrencySymbol(currencyCode) {
    if (!currencyCode) return '$';
    try {
        const parts = new Intl.NumberFormat('en-US', { style: 'currency', currency: currencyCode }).formatToParts(0);
        const symbol = parts.find(p => p.type === 'currency')?.value;
        return symbol || '$';
    } catch {
        return '$';
    }
}

// DOM Elements
const tickerInput = document.getElementById('ticker-input');
const autocompleteResults = document.getElementById('autocomplete-results');
const modelSelect = document.getElementById('model-select');
const modelDropdownTrigger = document.getElementById('model-dropdown-trigger');
const modelDropdownList = document.getElementById('model-dropdown-list');
const modelSelectedText = document.getElementById('model-selected-text');
const trainBtn = document.getElementById('train-btn');
const predictBtn = document.getElementById('predict-btn');
const downloadBtn = document.getElementById('download-btn');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const forecastList = document.getElementById('forecast-list');
const trendIndicator = document.getElementById('trend-indicator');
const trendText = document.getElementById('trend-text');

const newsFeedContainer = document.getElementById('news-feed-container');
const newsSentimentBadge = document.getElementById('news-sentiment-badge');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('stockChart')) {
        initChart([], []);
        fetchAndDisplayData(tickerInput ? tickerInput.value.trim() || 'INR=X' : 'INR=X');
    }
});

// Autocomplete Logic
let debounceTimeout;

if (tickerInput) {
    tickerInput.addEventListener('input', (e) => {
        const q = e.target.value.trim();
        clearTimeout(debounceTimeout);

        if (q.length < 1) {
            autocompleteResults.classList.add('hidden');
            return;
        }

        debounceTimeout = setTimeout(async () => {
            try {
                const resp = await fetch(`${API_BASE_URL}/search?q=${encodeURIComponent(q)}`);
                const data = await resp.json();

                if (data.quotes && data.quotes.length > 0) {
                    autocompleteResults.innerHTML = '';
                    data.quotes.forEach(quote => {
                        const item = document.createElement('div');
                        item.className = 'flex items-center justify-between p-3 hover:bg-surface-container-highest cursor-pointer rounded-lg transition-colors border-b border-outline-variant/10 last:border-0';
                        item.innerHTML = `
                            <div class="flex flex-col gap-1">
                                <span class="font-bold text-primary-container">${quote.symbol}</span>
                                <span class="text-on-surface-variant text-xs truncate max-w-[150px]">${quote.shortname || ''}</span>
                            </div>
                            <span class="text-[9px] bg-surface-container-highest px-2 py-0.5 rounded text-outline uppercase font-bold tracking-widest">${quote.type}</span>
                        `;
                        item.addEventListener('click', () => {
                            tickerInput.value = quote.symbol;
                            autocompleteResults.classList.add('hidden');
                        });
                        autocompleteResults.appendChild(item);
                    });
                    autocompleteResults.classList.remove('hidden');
                } else {
                    autocompleteResults.classList.add('hidden');
                }
            } catch (err) {
                console.error('Search error:', err);
            }
        }, 300);
    });
}

// Hide dropdown if clicked outside
document.addEventListener('click', (e) => {
    if (autocompleteResults && !e.target.closest('.relative.mb-6') && !e.target.closest('.input-box')) {
        autocompleteResults.classList.add('hidden');
    }
    if (modelDropdownList && !e.target.closest('#model-dropdown-trigger')) {
        modelDropdownList.classList.add('hidden');
    }
});

// Custom Inference Engine Dropdown Logic
if (modelDropdownTrigger && modelDropdownList) {
    modelDropdownTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        modelDropdownList.classList.toggle('hidden');
    });

    const options = modelDropdownList.querySelectorAll('li');
    options.forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            const val = option.getAttribute('data-value');
            // Select the text from the interior div/span
            const text = option.querySelector('span') ? option.querySelector('span').innerText : option.innerText;
            if (modelSelect) modelSelect.value = val;
            if (modelSelectedText) modelSelectedText.innerText = text;
            modelDropdownList.classList.add('hidden');
        });
    });
}

if (trainBtn) {
    trainBtn.addEventListener('click', async () => {
        const ticker = tickerInput.value.trim().toUpperCase();
        if (!ticker) return showStatus('Please enter a valid asset ticker.', true);

        showStatus(`Initializing neural engine for ${ticker}...`, false);

        try {
            const response = await fetch(`${API_BASE_URL}/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: ticker })
            });

            const data = await response.json();
            if (response.ok) {
                showStatus(data.message, false, 5000);
            } else {
                showStatus(data.detail, true);
            }
        } catch (error) {
            showStatus(`Network alignment failed: ${error.message}`, true);
        }
    });
}

if (predictBtn) {
    predictBtn.addEventListener('click', async () => {
        const ticker = tickerInput.value.trim().toUpperCase();
        const model = modelSelect ? modelSelect.value : 'lstm';

        if (!ticker) return showStatus('Please enter an asset ticker.', true);

        showStatus(`Running ${model.toUpperCase()} projection matrix on ${ticker}...`);
        predictBtn.disabled = true;

        try {
            await fetchAndDisplayData(ticker);
            fetchNewsAndSentiment(ticker);

            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker, model_type: model, prediction_days: 5 })
            });

            const data = await response.json();

            if (response.ok) {
                showStatus('Projection synchronized successfully.', false, 3000);
                displayPredictions(data.predictions);
                addPredictionsToChart(data.predictions);
                if (downloadBtn) downloadBtn.disabled = false;
            } else {
                showStatus(data.detail, true);
            }
        } catch (error) {
            showStatus(`Projection error: ${error.message}`, true);
        } finally {
            predictBtn.disabled = false;
        }
    });
}

async function fetchAndDisplayData(ticker) {
    try {
        const response = await fetch(`${API_BASE_URL}/stock-data?ticker=${ticker}`);
        const data = await response.json();

        if (response.ok) {
            currentCurrencySymbol = getCurrencySymbol(data.currency);

            // Reconfigure existing chart dynamically
            if (stockChart) {
                stockChart.options.plugins.tooltip.callbacks.label = (ctx) => currentCurrencySymbol + ctx.parsed.y.toFixed(2);
                stockChart.options.scales.y.ticks.callback = (val) => currentCurrencySymbol + val;
            }

            const labels = data.data.map(item => item.Date);
            const prices = data.data.map(item => item.Close);
            updateChart(labels, prices, ticker);

            // Update Terminal Asset Info
            const assetTickerInfo = document.getElementById('asset-ticker');
            if (assetTickerInfo) assetTickerInfo.textContent = ticker;

            const assetPriceInfo = document.getElementById('asset-price');
            if (assetPriceInfo && prices.length > 0) {
                assetPriceInfo.textContent = currentCurrencySymbol + prices[prices.length - 1].toFixed(2);
            }

        } else {
            console.error('Failed to align historical sequences.');
        }
    } catch (err) {
        console.error('Data pull error:', err);
    }
}

async function fetchNewsAndSentiment(ticker) {
    try {
        const response = await fetch(`${API_BASE_URL}/news-sentiment?ticker=${ticker}`);
        const data = await response.json();

        if (response.ok) {
            displayNews(data);
        } else {
            console.error('Failed to align news analysis.');
        }
    } catch (err) {
        console.error('News pull error:', err);
    }
}

function displayNews(data) {
    const { news, overall_label, overall_sentiment } = data;

    if (news.length === 0) {
        if (newsFeedContainer) {
            newsFeedContainer.innerHTML = '<div class="p-6 text-on-surface-variant flex items-center gap-3"><i class="ph-light ph-newspaper text-xl"></i><span>No recent intelligence found for this asset.</span></div>';
        }
        return;
    }

    if (newsSentimentBadge) {
        newsSentimentBadge.classList.remove('hidden');
        const badgeContent = document.getElementById('news-sentiment-text');
        const badgeDot = newsSentimentBadge.querySelector('.w-2.h-2.rounded-full');

        if (overall_label === 'Bullish' || overall_label === 'Positive') {
            newsSentimentBadge.className = 'flex items-center gap-2 bg-tertiary-container/10 border border-tertiary-container/30 px-3 py-1.5 rounded-full';
            if (badgeDot) badgeDot.className = 'w-2 h-2 rounded-full bg-tertiary shadow-[0_0_8px_#49ed72]';
            if (badgeContent) badgeContent.className = 'text-[10px] font-bold uppercase tracking-widest text-tertiary-container';
            if (badgeContent) badgeContent.textContent = 'Bullish Sentiment';
        } else if (overall_label === 'Bearish' || overall_label === 'Negative') {
            newsSentimentBadge.className = 'flex items-center gap-2 bg-error/10 border border-error/30 px-3 py-1.5 rounded-full';
            if (badgeDot) badgeDot.className = 'w-2 h-2 rounded-full bg-error shadow-[0_0_8px_#ffb4ab]';
            if (badgeContent) badgeContent.className = 'text-[10px] font-bold uppercase tracking-widest text-error';
            if (badgeContent) badgeContent.textContent = 'Bearish Sentiment';
        } else {
            newsSentimentBadge.className = 'flex items-center gap-2 bg-surface-bright border border-outline-variant/30 px-3 py-1.5 rounded-full';
            if (badgeDot) badgeDot.className = 'w-2 h-2 rounded-full bg-outline-variant';
            if (badgeContent) badgeContent.className = 'text-[10px] font-bold uppercase tracking-widest text-on-surface-variant';
            if (badgeContent) badgeContent.textContent = 'Neutral Sentiment';
        }
    }

    if (newsFeedContainer) {
        newsFeedContainer.innerHTML = '';
        news.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'p-6 hover:bg-white/5 transition-colors group cursor-pointer border-b border-white/5 last:border-0';

            const labelClass = item.sentiment_label.toLowerCase();
            let icon = 'ph-minus';
            let colorClass = 'text-on-surface-variant';

            let impactEst = (item.sentiment_score * 2.5).toFixed(2);
            let impactSign = item.sentiment_score > 0 ? '+' : '';
            if (item.sentiment_score === 0) impactSign = '±';
            let impactText = `${impactSign}${Math.abs(impactEst)}%`;

            if (labelClass === 'positive') {
                icon = 'ph-trend-up';
                colorClass = 'text-tertiary-container';
            } else if (labelClass === 'negative') {
                icon = 'ph-trend-down';
                colorClass = 'text-error';
            }

            card.innerHTML = `
                <div class="flex justify-between items-start mb-2">
                    <span class="bg-surface-bright text-on-surface-variant px-2 py-0.5 rounded text-[10px] font-bold uppercase">${item.publisher}</span>
                    <div class="flex items-center gap-3">
                        <span class="${colorClass} text-[10px] font-bold uppercase tracking-wider bg-black/20 px-2 py-0.5 rounded border border-white/5">
                            Est. Impact: ${impactText}
                        </span>
                        <span class="${colorClass} text-[10px] font-bold flex items-center gap-1 uppercase">
                            <i class="ph-bold ${icon}"></i> ${item.sentiment_label}
                        </span>
                    </div>
                </div>
                <h4 class="text-on-surface text-sm font-semibold group-hover:text-primary transition-colors mb-2"><a href="${item.link}" target="_blank">${item.title}</a></h4>
                ${item.summary ? `<p class="text-xs text-on-surface-variant/80 line-clamp-2">${item.summary}</p>` : ''}
            `;
            newsFeedContainer.appendChild(card);
        });
    }
}

// UI Functionality
function showStatus(message, isError = false, timeout = 0) {
    if (!statusBar) return;

    statusBar.classList.remove('hidden', 'error');
    statusText.textContent = message;

    if (isError) {
        statusBar.classList.add('error');
        if (timeout === 0) timeout = 5000;
    }

    if (timeout > 0) {
        setTimeout(() => statusBar.classList.add('hidden'), timeout);
    }
}

function displayPredictions(predictions) {
    if (forecastList) {
        forecastList.innerHTML = '';
        predictions.forEach((pred, index) => {
            const item = document.createElement('div');
            item.className = 'forecast-card p-4 rounded-xl border border-outline-variant/30 bg-[#15171b] flex justify-between items-center hover:border-primary/50 transition-colors group relative overflow-hidden';
            item.style.animationDelay = `${index * 0.1}s`;

            const dateObj = new Date(pred.date);
            const dayName = dateObj.toLocaleDateString('en-US', { weekday: 'short' }).toUpperCase();
            const fullDate = dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

            item.innerHTML = `
                <div class="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div class="flex flex-col relative z-10">
                    <span class="text-[10px] uppercase tracking-[0.1em] text-on-surface-variant font-bold mb-1 group-hover:text-primary transition-colors">DAY ${index + 1} • ${dayName}</span>
                    <span class="text-sm font-headline font-bold text-on-surface fc-full">${fullDate}</span>
                </div>
                <div class="text-xl font-headline font-black text-primary relative z-10 fc-price">${currentCurrencySymbol}${pred.predicted_close.toFixed(2)}</div>
            `;
            forecastList.appendChild(item);
        });
    }

    // Update Trend Badge
    let prevPrice = stockChart.data.datasets[0].data[stockChart.data.datasets[0].data.length - 1];
    const lastPredPrice = predictions[predictions.length - 1].predicted_close;

    if (trendIndicator) {
        trendIndicator.classList.remove('hidden');

        if (lastPredPrice > prevPrice) {
            trendIndicator.className = 'flex items-center gap-1 bg-tertiary-container/10 border border-tertiary-container/30 px-3 py-1.5 rounded-full text-xs font-bold text-tertiary-container';
            trendIndicator.innerHTML = '<i class="ph-bold ph-trend-up"></i><span id="trend-text">Bullish Outlook</span>';
        } else {
            trendIndicator.className = 'flex items-center gap-1 bg-error/10 border border-error/30 px-3 py-1.5 rounded-full text-xs font-bold text-error';
            trendIndicator.innerHTML = '<i class="ph-bold ph-trend-down"></i><span id="trend-text">Bearish Outlook</span>';
        }
    }
}

// Chart Management
function initChart(labels, data) {
    const ctx = document.getElementById('stockChart').getContext('2d');

    // Global Options
    Chart.defaults.color = '#8a8d9b';
    Chart.defaults.font.family = "'Inter', sans-serif";

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Historical Asset Value',
                data: data,
                borderColor: '#00e5ff',
                backgroundColor: 'rgba(0, 229, 255, 0.05)',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(26, 28, 32, 0.95)',
                    titleFont: { family: "'Manrope', sans-serif", size: 14, weight: 'bold' },
                    bodyFont: { family: "'Manrope', sans-serif", size: 14 },
                    titleColor: '#bac9cc',
                    bodyColor: '#00e5ff',
                    padding: 16,
                    borderWidth: 1,
                    borderColor: 'rgba(255,255,255,0.1)',
                    displayColors: false,
                    callbacks: { label: (ctx) => currentCurrencySymbol + ctx.parsed.y.toFixed(2) }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.02)', drawBorder: false },
                    ticks: { maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.02)', drawBorder: false },
                    beginAtZero: false,
                    ticks: { callback: (val) => currentCurrencySymbol + val }
                }
            }
        }
    });
}

function updateChart(labels, data, ticker) {
    const dLabels = labels.slice(-100);
    const dData = data.slice(-100);

    stockChart.data.labels = dLabels;
    stockChart.data.datasets = [{
        label: `${ticker} historical`,
        data: dData,
        borderColor: '#00e5ff',
        backgroundColor: 'rgba(0, 229, 255, 0.05)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.15
    }];

    const minVal = Math.min(...dData);
    const maxVal = Math.max(...dData);
    stockChart.options.scales.y.min = minVal * 0.98;
    stockChart.options.scales.y.max = maxVal * 1.02;
    stockChart.options.scales.y.suggestedMin = undefined;
    stockChart.options.scales.y.suggestedMax = undefined;

    stockChart.update();
}

function addPredictionsToChart(predictions) {
    const predDates = predictions.map(p => p.date);
    const predPrices = predictions.map(p => p.predicted_close);

    const lastHistPrice = stockChart.data.datasets[0].data[stockChart.data.datasets[0].data.length - 1];
    const combinedLabels = [...stockChart.data.labels, ...predDates];

    const predData = Array(stockChart.data.labels.length - 1).fill(null);
    predData.push(lastHistPrice);
    predData.push(...predPrices);

    if (stockChart.data.datasets.length > 1) {
        stockChart.data.datasets[1].data = predData;
        stockChart.data.labels = combinedLabels;
    } else {
        stockChart.data.labels = combinedLabels;
        stockChart.data.datasets.push({
            label: 'Neural Forecast',
            data: predData,
            borderColor: '#f9abff', // secondary brand color from Tailwind
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [8, 8],
            pointRadius: 4,
            pointBackgroundColor: '#f9abff',
            pointBorderColor: '#0a0a0c',
            pointBorderWidth: 2
        });
    }

    const allVisible = [...stockChart.data.datasets[0].data, ...predPrices];
    const minVal = Math.min(...allVisible);
    const maxVal = Math.max(...allVisible);
    stockChart.options.scales.y.min = minVal * 0.98;
    stockChart.options.scales.y.max = maxVal * 1.02;

    stockChart.update();
}

// PDF Export
if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
        try {
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();

            const ticker = tickerInput ? tickerInput.value.toUpperCase() : 'ASSET';
            const model = modelSelectedText ? modelSelectedText.innerText : 'Neural Engine';

            doc.setFillColor(10, 10, 12);
            doc.rect(0, 0, 210, 297, "F");

            doc.setFontSize(22);
            doc.setTextColor(0, 229, 255); // Cyan brand
            doc.text("Intelligence Prediction Report", 20, 30);

            doc.setFontSize(11);
            doc.setTextColor(138, 141, 155);
            doc.text(`Asset Identifier: ${ticker}`, 20, 45);
            doc.text(`Neural Engine: ${model}`, 20, 52);
            doc.text(`Generation Date: ${new Date().toLocaleDateString()}`, 20, 59);

            doc.setFontSize(14);
            doc.setTextColor(255, 255, 255);
            doc.text("5-Day Trajectory Projection:", 20, 80);

            const items = document.querySelectorAll('.forecast-card');
            let startY = 90;

            if (items.length > 0) {
                items.forEach((item, index) => {
                    const fullDate = item.querySelector('.fc-full').textContent;
                    const price = item.querySelector('.fc-price').textContent;
                    doc.text(`-- }  (${fullDate}): ${price}`, 25, startY);
                    startY += 12;
                });
            }

            const canvas = document.getElementById('stockChart');
            if (canvas) {
                const chartImage = canvas.toDataURL('image/png', 1.0);
                doc.addImage(chartImage, 'PNG', 15, startY + 10, 180, 80);
            }

            doc.save(`${ticker}_Report.pdf`);
            showStatus('PDF Export Complete!', false, 3000);
        } catch (err) {
            console.error('PDF Export failed:', err);
            showStatus('Export failed: ' + err.message, true, 5000);
        }
    });
}

// Generic Button Handlers for Models and Docs pages
const viewWeightsBtn = document.getElementById('view-weights-btn');
const shapValuesBtn = document.getElementById('shap-values-btn');
const viewDictMapBtn = document.getElementById('view-dict-map-btn');

if (viewWeightsBtn) {
    viewWeightsBtn.addEventListener('click', (e) => {
        e.preventDefault();
        showStatus('Extracting neural weights... Just kidding, feature coming soon!', false, 3000);
    });
}

if (shapValuesBtn) {
    shapValuesBtn.addEventListener('click', (e) => {
        e.preventDefault();
        showStatus('Calculating SHAP values... Visualizer in development.', false, 3000);
    });
}

if (viewDictMapBtn) {
    viewDictMapBtn.addEventListener('click', (e) => {
        e.preventDefault();
        showStatus('Loading Semantic Dictionary Map... Interface coming soon.', false, 3000);
    });
}
