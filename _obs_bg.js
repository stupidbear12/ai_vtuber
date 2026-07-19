const WebSocket = require('ws');
const crypto = require('crypto');
const ws = new WebSocket('ws://localhost:4455');
ws.on('message', (data) => {
    const msg = JSON.parse(data);
    if (msg.op === 0) {
        const h1 = crypto.createHash('sha256').update('LzDko8s5VoYj3XoA' + msg.d.authentication.salt).digest('base64');
        const auth = crypto.createHash('sha256').update(h1 + msg.d.authentication.challenge).digest('base64');
        ws.send(JSON.stringify({op:1,d:{rpcVersion:1,authentication:auth}}));
    } else if (msg.op === 2) {
        ws.send(JSON.stringify({op:6,d:{requestType:'SetCurrentProgramScene',requestId:'sw',requestData:{sceneName:'Radio Mode'}}}));
    } else if (msg.op === 7) { ws.close(); }
});
ws.on('close', () => process.exit(0));
setTimeout(() => process.exit(0), 5000);
