// WebSocket client for live dashboard updates
(function() {
    let ws = null;
    let reconnectDelay = 1000;
    const maxReconnectDelay = 30000;
    const statusDot = document.getElementById('ws-status');

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws`;

        ws = new WebSocket(url);

        ws.onopen = function() {
            reconnectDelay = 1000;
            if (statusDot) {
                statusDot.className = 'w-2 h-2 rounded-full bg-green-500';
                statusDot.title = 'Connected';
            }
            // Send ping every 30s to keep alive
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send('ping');
                }
            }, 30000);
        };

        ws.onmessage = function(event) {
            try {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            } catch (e) {
                // ignore parse errors
            }
        };

        ws.onclose = function() {
            if (statusDot) {
                statusDot.className = 'w-2 h-2 rounded-full bg-red-500';
                statusDot.title = 'Disconnected';
            }
            setTimeout(() => {
                reconnectDelay = Math.min(reconnectDelay * 2, maxReconnectDelay);
                connect();
            }, reconnectDelay);
        };

        ws.onerror = function() {
            ws.close();
        };
    }

    function handleMessage(msg) {
        switch (msg.type) {
            case 'trade_update':
                updateTradeCard(msg);
                break;
            case 'trade_opened':
            case 'trade_closed':
            case 'portfolio_update':
                // Reload page for structural changes
                location.reload();
                break;
            case 'log_entry':
                appendLogEntry(msg);
                break;
            case 'pong':
                break;
        }
    }

    function updateTradeCard(msg) {
        // Update daily P&L display if provided
        if (msg.daily_pnl !== undefined) {
            const el = document.getElementById('daily-pnl');
            if (el) {
                const val = msg.daily_pnl;
                const sign = val >= 0 ? '+' : '';
                el.textContent = `${sign}$${val.toFixed(2)}`;
                el.className = el.className.replace(/text-(green|red)-400/g, '');
                el.classList.add(val >= 0 ? 'text-green-400' : 'text-red-400');
            }
        }
    }

    function appendLogEntry(msg) {
        const container = document.getElementById('error-log');
        if (!container) return;

        // Remove "no errors" placeholder if present
        const placeholder = container.querySelector('.text-center');
        if (placeholder) placeholder.remove();

        const div = document.createElement('div');
        div.className = `py-1 border-b border-owl-border/50 ${msg.level === 'ERROR' ? 'text-red-400' : 'text-yellow-400'}`;
        div.innerHTML = `<span class="text-gray-500">${msg.timestamp || ''}</span> <span class="font-bold">${msg.level}</span> ${msg.message}`;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    // Auto-connect
    if (document.getElementById('ws-status')) {
        connect();
    }
})();
